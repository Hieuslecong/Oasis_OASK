"""OASIS-RC-v2.1 scientific training runner.

This entrypoint is intentionally separate from the reconstructed v2.0.4 runner.
It keeps the canonical test closed and supports development/confirmatory arms with
explicit optimizer and auxiliary-loss contracts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    sha256_file,
    validate_critic_checkpoint,
)
from oasis_rc_v2.corruptions import build_targets, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.energy_qualification import summarize_energy_trajectory
from oasis_rc_v2.losses import (
    adversarial_pair_student_loss,
    continuous_relation_path_loss,
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from oasis_rc_v2.protocol import verify_gate0_certificate
from oasis_rc_v2.qualification import connected_gate_failures
from .aosk import oriented_consistency_loss
from .topology_loss import centerline_cldice_loss
from .train_oasis_rc_v2 import (
    configure_determinism,
    load_student_init,
    make_generator,
    make_loader,
    make_student,
    make_train_loader,
    manifest_has_split,
    manifest_splits,
    runtime_metadata,
    seed_all,
    select_threshold,
)

ARMS = {
    "control",
    "connected",
    "aosk",
    "aosk_connected",
    "cldice",
    "adversarial",
}
AOSK_VARIANT = "oriented-consistency-v1-isotropic-flat"


def _optimizer(parameters, args):
    return torch.optim.AdamW(
        parameters,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(float(args.beta1), float(args.beta2)),
        eps=float(args.adam_eps),
    )


def _unpack(batch):
    if len(batch) == 3:
        return batch
    x, y = batch
    return x, y, y.flatten(1).sum(1) == 0


def _critic_hparams(args, cfg, determinism_mode):
    return {
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "betas": [float(args.beta1), float(args.beta2)],
        "adam_eps": float(args.adam_eps),
        "critic_epochs": int(args.critic_epochs),
        "critic_width": int(args.critic_width),
        "batch_size": int(cfg["batch_size"]),
        "crack_dice_weight": float(args.crack_dice_weight),
        "mismatch_weight": float(args.mismatch_weight),
        "pair_weight": float(args.pair_weight),
        "normal_critic_weight": float(args.normal_critic_weight),
        "normal_fraction": float(args.normal_fraction),
        "path_weight": float(args.path_weight),
        "path_margin": float(args.path_margin),
        "path_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        "determinism_mode": determinism_mode,
        "method_spec": "METHOD_SPEC_V2_1.md",
    }


def train_critic(args, cfg, device, out, determinism_mode):
    loader, sampler = make_train_loader(
        args.manifest, cfg["image_size"], cfg["batch_size"],
        args.normal_fraction, int(cfg["seed"]), cfg.get("num_workers", 0)
    )
    critic = OASISRCv2Critic(width=args.critic_width).to(device)
    optimizer = _optimizer(critic.parameters(), args)
    corruption_generator = make_generator(device, int(cfg["seed"]) + 30002)
    history = []
    for epoch in range(args.critic_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        critic.train()
        rows = []
        for batch in loader:
            x, y, is_normal = _unpack(batch)
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            wrong, invalid = make_corrupted_mask(
                y, true_normal=is_normal, generator=corruption_generator, image=x
            )
            semantic, mismatch, pair_valid = build_targets(wrong, invalid)
            class_loss, terms = oasis_rc_critic_loss(
                critic(x, wrong), semantic, mismatch, pair_valid,
                crack_dice_weight=args.crack_dice_weight,
                mismatch_weight=args.mismatch_weight,
                pair_weight=args.pair_weight,
            )
            path_loss, path_terms = continuous_relation_path_loss(
                critic, x, y, wrong,
                pair_weight=args.student_pair_weight,
                margin=args.path_margin,
            )
            loss = class_loss + float(args.path_weight) * path_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(args.grad_clip))
            optimizer.step()
            rows.append({
                "loss": float(loss.detach()),
                "classification": float(class_loss.detach()),
                "path": float(path_loss.detach()),
                "path_order_fraction": float(path_terms["path_order_fraction"]),
                "path_pairs": int(path_terms["path_pairs"]),
                **{f"critic_{k}": float(v) for k, v in terms.items() if torch.is_tensor(v)},
            })
        history.append({
            "epoch": epoch,
            "loss": sum(r["loss"] for r in rows) / max(1, len(rows)),
            "path": sum(r["path"] for r in rows) / max(1, len(rows)),
            "path_order_fraction": sum(r["path_order_fraction"] for r in rows) / max(1, len(rows)),
        })
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "seed": int(cfg["seed"]),
        "critic": critic.state_dict(),
        "width": int(args.critic_width),
        "config": dict(cfg),
        "manifest_file_sha256": sha256_file(args.manifest),
        "dataset_content_sha256": args._dataset_content_sha256,
        "full_gate0_certificate_sha256": sha256_file(args.full_gate0_certificate),
        "normal_fraction": float(args.normal_fraction),
        "normal_critic_weight": float(args.normal_critic_weight),
        "training_hparams": _critic_hparams(args, cfg, determinism_mode),
        "runtime": runtime_metadata(device, determinism_mode),
    }
    torch.save(checkpoint, out / "critic.pt")
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic


@torch.no_grad()
def energy_qualification(critic, loader, device, pair_weight, margin):
    critic.eval()
    trajectories = []
    generator = make_generator(device, 99173)
    for batch in loader:
        x, y, is_normal = _unpack(batch)
        x, y = x.to(device), y.to(device)
        is_normal = is_normal.to(device, dtype=torch.bool)
        wrong, _ = make_corrupted_mask(y, true_normal=is_normal, generator=generator, image=x)
        rows = summarize_energy_trajectory(
            critic, x, y, wrong, pair_weight=pair_weight,
            levels=(0.0, 0.25, 0.5, 0.75, 1.0), margin=margin,
        )
        trajectories.append(rows)
    if not trajectories:
        return {"energy_samples": 0, "energy_finite": False}
    keys = set().union(*(r.keys() for r in trajectories))
    result = {}
    weighted_keys = {
        "positive_energy_gap_fraction", "continuous_path_order_fraction",
        "mean_energy_gap", "median_energy_gap",
    }
    total_samples = sum(int(r.get("energy_samples", 0)) for r in trajectories)
    for key in weighted_keys:
        values = [(float(r[key]), int(r.get("energy_samples", 0))) for r in trajectories if r.get(key) is not None]
        if values:
            result[key] = sum(v*n for v, n in values) / max(1, sum(n for _, n in values))
    result["energy_samples"] = total_samples
    result["energy_finite"] = all(bool(r.get("energy_finite", False)) for r in trajectories)
    return result


def _validate_loaded_critic(saved, args, cfg):
    expected = _critic_hparams(args, cfg, args._determinism_mode)
    return validate_critic_checkpoint(
        saved, args.manifest, cfg, args.normal_fraction, args.normal_critic_weight,
        dataset_content_sha256_value=args._dataset_content_sha256,
        expected_hparams=expected,
        full_gate0_certificate=args.full_gate0_certificate,
    )


def train_student(args, cfg, device, out, critic=None):
    seed = int(cfg["seed"])
    seed_all(seed)
    loader, sampler = make_train_loader(
        args.manifest, cfg["image_size"], cfg["batch_size"],
        args.normal_fraction, seed, cfg.get("num_workers", 0)
    )
    val_loader = make_loader(
        args.manifest, "val", cfg["image_size"], cfg["batch_size"], False,
        cfg.get("num_workers", 0), seed=seed
    )
    student = make_student(args.student_kind, args.student_width).to(device)
    setattr(student, "_oasis_width", int(args.student_width))
    load_student_init(student, args.student_init_checkpoint, seed)
    optimizer = _optimizer(student.parameters(), args)
    generator = make_generator(device, seed + 20001)
    if critic is not None:
        critic.eval()
        for p in critic.parameters():
            p.requires_grad_(False)
    best_key = None
    best_state = None
    history = []
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        student.train()
        epoch_rows = []
        for batch in loader:
            x, y, is_normal = _unpack(batch)
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            logits = student(x)
            seg = segmentation_loss(logits, y)
            total = seg
            aux = logits.new_zeros(())
            if args.mode in {"connected", "aosk_connected"}:
                prediction = logits.sigmoid()
                wrong, _ = make_corrupted_mask(y, true_normal=is_normal, generator=generator, image=x)
                with torch.no_grad():
                    gt_out = critic(x, y)
                    corrupt_out = critic(x, wrong)
                aux, _ = oasis_rc_student_loss_v2(
                    critic(x, prediction), gt_out, corrupt_out, prediction, y,
                    margin=args.student_margin,
                    pair_weight=args.student_pair_weight,
                    corrupted_rank_weight=args.corrupted_rank_weight,
                    fp_weight=args.fp_weight,
                )
                total = total + float(args.lambda_oasis) * aux
            elif args.mode == "adversarial":
                prediction = logits.sigmoid()
                aux = adversarial_pair_student_loss(critic(x, prediction))
                total = total + float(args.lambda_adversarial) * aux
            if args.mode in {"aosk", "aosk_connected"}:
                a = oriented_consistency_loss(logits, x, y)
                total = total + float(args.lambda_aosk) * a
            if args.mode == "cldice":
                c = centerline_cldice_loss(logits, y)
                total = total + float(args.lambda_cldice) * c
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_clip))
            optimizer.step()
            epoch_rows.append((float(total.detach()), float(seg.detach()), float(aux.detach())))
        validation = select_threshold(student, val_loader, device)
        key = float(validation["dice"])
        history.append({
            "epoch": epoch,
            "loss": sum(r[0] for r in epoch_rows)/max(1,len(epoch_rows)),
            "seg": sum(r[1] for r in epoch_rows)/max(1,len(epoch_rows)),
            "aux": sum(r[2] for r in epoch_rows)/max(1,len(epoch_rows)),
            "val": validation,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_state = {k: v.detach().cpu().clone() for k,v in student.state_dict().items()}
    student.load_state_dict(best_state)
    validation = select_threshold(student, val_loader, device)
    effective = {
        "method_version": METHOD_VERSION,
        "seed": seed,
        "image_size": int(cfg["image_size"]),
        "batch_size": int(cfg["batch_size"]),
        "epochs": int(args.epochs),
        "mode": args.mode,
        "student_kind": args.student_kind,
        "student_width": int(args.student_width),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "betas": [float(args.beta1), float(args.beta2)],
        "adam_eps": float(args.adam_eps),
        "grad_clip": float(args.grad_clip),
        "student_margin": float(args.student_margin),
        "student_pair_weight": float(args.student_pair_weight),
        "corrupted_rank_weight": float(args.corrupted_rank_weight),
        "fp_weight": float(args.fp_weight),
        "lambda_oasis": float(args.lambda_oasis),
        "lambda_aosk": float(args.lambda_aosk),
        "lambda_cldice": float(args.lambda_cldice),
        "lambda_adversarial": float(args.lambda_adversarial),
        "normal_fraction": float(args.normal_fraction),
        "checkpoint_selection": "max-validation-micro-dice",
        "threshold_selection": "max-validation-micro-dice-normal-fp-tiebreak",
    }
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "seed": seed,
        "student": student.state_dict(),
        "student_kind": args.student_kind,
        "student_width": int(args.student_width),
        "mode": args.mode,
        "effective_config": effective,
        "manifest_file_sha256": sha256_file(args.manifest),
        "dataset_content_sha256": args._dataset_content_sha256,
        "training_view_dataset_sha256": args._dataset_content_sha256,
        "gate0_certificate_sha256": sha256_file(args.gate0_certificate),
        "full_gate0_certificate_sha256": sha256_file(args.full_gate0_certificate),
        "student_init_sha256": sha256_file(args.student_init_checkpoint),
        "critic_checkpoint_sha256": sha256_file(args.critic_checkpoint),
        "threshold_validation": float(validation["threshold"]),
        "runtime": runtime_metadata(device, args._determinism_mode),
        "inference_contract": "RGB -> crack logits only",
    }
    torch.save(checkpoint, out / "student_only.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    (out / "effective_config.json").write_text(json.dumps(effective, indent=2))
    return student, validation


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--manifest", required=True)
    p.add_argument("--gate0-certificate", required=True); p.add_argument("--full-gate0-certificate", required=True)
    p.add_argument("--out", required=True); p.add_argument("--mode", choices=sorted(ARMS | {"critic"}), required=True)
    p.add_argument("--student-init-checkpoint"); p.add_argument("--critic-checkpoint")
    p.add_argument("--student-kind", default="mobilenetv3"); p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--normal-fraction", type=float, default=0.0); p.add_argument("--normal-critic-weight", type=float, default=1.0)
    p.add_argument("--critic-epochs", type=int, default=10); p.add_argument("--epochs", type=int, default=12); p.add_argument("--critic-width", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4); p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9); p.add_argument("--beta2", type=float, default=0.999); p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=5.0); p.add_argument("--crack-dice-weight", type=float, default=1.0)
    p.add_argument("--mismatch-weight", type=float, default=1.0); p.add_argument("--pair-weight", type=float, default=0.25)
    p.add_argument("--path-weight", type=float, default=0.25); p.add_argument("--path-margin", type=float, default=0.02)
    p.add_argument("--student-margin", type=float, default=0.10); p.add_argument("--student-pair-weight", type=float, default=0.25)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0); p.add_argument("--fp-weight", type=float, default=1.0)
    p.add_argument("--lambda-oasis", type=float, default=0.001); p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--lambda-cldice", type=float, default=0.1); p.add_argument("--lambda-adversarial", type=float, default=0.001)
    p.add_argument("--determinism-mode", choices=("off","best_effort","strict"), default="strict")
    return p


def main():
    args = parser().parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("seed","image_size","batch_size","device"):
        if key not in cfg: raise ValueError(f"config missing {key}")
    if "test" in manifest_splits(args.manifest): raise ValueError("v2.1 trainer refuses canonical test rows")
    cert = verify_gate0_certificate(
        args.gate0_certificate, args.manifest, int(cfg["image_size"]),
        "train" if args.normal_fraction > 0 else "none", args.full_gate0_certificate,
    )
    args._dataset_content_sha256 = cert["dataset_content_sha256"]
    args._determinism_mode = args.determinism_mode
    device = torch.device(cfg["device"]); seed_all(cfg["seed"]); configure_determinism(args.determinism_mode, device.type)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    critic = None
    if args.mode == "critic":
        critic = train_critic(args,cfg,device,out,args.determinism_mode)
    elif args.mode in {"connected","aosk_connected","adversarial"}:
        if not args.critic_checkpoint: raise ValueError(f"{args.mode} requires --critic-checkpoint")
        saved = torch.load(args.critic_checkpoint,map_location=device,weights_only=False); _validate_loaded_critic(saved,args,cfg)
        critic = OASISRCv2Critic(width=int(saved["width"])).to(device); critic.load_state_dict(saved["critic"])
    if critic is not None:
        val = make_loader(args.manifest,"val",cfg["image_size"],cfg["batch_size"],False,cfg.get("num_workers",0),seed=int(cfg["seed"]),return_is_normal=True)
        metrics = energy_qualification(critic,val,device,args.student_pair_weight,args.path_margin)
        (out/"critic_energy_validation.json").write_text(json.dumps(metrics,indent=2))
        failures = connected_gate_failures(metrics, classification_metrics=None, require_classification=False)
        if failures and args.mode != "critic": raise RuntimeError("v2.1 energy gate failed: "+", ".join(failures))
    if args.mode == "critic": return
    if not args.student_init_checkpoint: raise ValueError("student runs require --student-init-checkpoint")
    return train_student(args,cfg,device,out,critic=critic)


if __name__ == "__main__": main()
