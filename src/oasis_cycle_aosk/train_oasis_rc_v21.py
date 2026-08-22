"""OASIS-RC-v2.1 real-data development runner.

This entrypoint is intentionally separate from reconstructed v2.0.4. It refuses
canonical/held-out test rows. Connected/frozen-pair arms require a schema-5
critic whose representation and dedicated relation-energy gates both pass.
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
    TRAINER_CONTRACT,
    sha256_file,
    validate_critic_checkpoint,
)
from oasis_rc_v2.corruptions import build_targets, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.energy_qualification import (
    gradient_alignment_diagnostics,
    relation_energy_trajectory,
    summarize_energy_trajectories,
)
from oasis_rc_v2.losses import (
    adversarial_pair_student_loss,
    continuous_relation_path_loss,
    critic_endpoint_energy_loss,
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from oasis_rc_v2.protocol import verify_gate0_certificate
from oasis_rc_v2.qualification import connected_gate_failures, relation_energy_gate_failures
from .aosk import oriented_consistency_loss
from .topology_loss import centerline_cldice_loss
from .train_oasis_rc_v2 import (
    augment,
    configure_determinism,
    critic_metrics,
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

ARMS = {"control", "connected", "aosk", "aosk_connected", "cldice", "adversarial"}
ENERGY_HEAD_CONTRACT = "dedicated-scalar-lower-is-better-v1"
AOSK_VARIANT = "structure-tensor-steered-v2"


def _optimizer(params, args):
    return torch.optim.AdamW(
        params,
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


def _mean(values):
    return sum(values) / max(1, len(values))


def _critic_hparams(args, cfg, determinism):
    return {
        # Historical optimizer settings are provenance only. The checkpoint
        # validator compares only its declared scientific compatibility subset.
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
        "rgb_mask_weight": float(args.rgb_mask_weight),
        "normal_critic_weight": float(args.normal_critic_weight),
        "normal_fraction": float(args.normal_fraction),
        "endpoint_weight": float(args.endpoint_weight),
        "endpoint_anchor_weight": float(args.endpoint_anchor_weight),
        "endpoint_margin": float(args.endpoint_margin),
        "path_weight": float(args.path_weight),
        "path_margin": float(args.path_margin),
        "path_levels": [0.0, 0.25, 0.5, 0.75, 1.0],
        "energy_head_contract": ENERGY_HEAD_CONTRACT,
        "determinism_mode": determinism,
        "rgb_shuffle_pair_only": True,
        "mask_flip_training": False,
        "mask_variant_contract": "operator-preserved-v1",
        "method_spec": "METHOD_SPEC_V2_1.md",
    }


def _critic_term(critic, x, mask, invalid, args):
    semantic, mismatch, pair_valid = build_targets(mask, invalid)
    return oasis_rc_critic_loss(
        critic(x, mask),
        semantic,
        mismatch,
        pair_valid,
        crack_dice_weight=args.crack_dice_weight,
        mismatch_weight=args.mismatch_weight,
        pair_weight=args.pair_weight,
    )[0]


def train_critic(args, cfg, device, out, determinism):
    loader, sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        int(cfg["seed"]),
        cfg.get("num_workers", 0),
    )
    critic = OASISRCv2Critic(width=args.critic_width).to(device)
    optimizer = _optimizer(critic.parameters(), args)
    aug_generator = make_generator(device, int(cfg["seed"]) + 30001)
    corruption_generator = make_generator(device, int(cfg["seed"]) + 30002)
    history = []

    for epoch in range(args.critic_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        critic.train()
        totals = []
        representation_losses = []
        endpoint_losses = []
        path_losses = []
        endpoint_orders = []
        path_orders = []
        endpoint_gaps = []
        normal_seen = 0
        rgb_seen = 0
        corruption_counts = {}

        for batch in loader:
            x, y, is_normal = _unpack(batch)
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, aug_generator)
            normal_seen += int(is_normal.sum())
            wrong, invalid, meta = make_corrupted_mask(
                y,
                true_normal=is_normal,
                generator=corruption_generator,
                image=x,
                return_meta=True,
            )
            for item in meta:
                corruption_counts[item["kind"]] = corruption_counts.get(item["kind"], 0) + 1

            clean_loss = _critic_term(critic, x, y, torch.zeros_like(y), args)
            corrupt_loss = _critic_term(critic, x, wrong, invalid, args)
            representation = 0.5 * (clean_loss + corrupt_loss)

            crack_rows = (~is_normal) & (y.flatten(1).sum(1) > 0)
            if crack_rows.any():
                xc, yc = x[crack_rows], y[crack_rows]
                rgb_seen += int(yc.shape[0])
                semantic, mismatch, pair_valid = build_targets(yc, torch.zeros_like(yc))
                rgb_pair_only, _ = oasis_rc_critic_loss(
                    critic(xc.flip(-1), yc),
                    semantic,
                    mismatch,
                    torch.zeros_like(pair_valid),
                    crack_dice_weight=args.crack_dice_weight,
                    mismatch_weight=args.mismatch_weight,
                    pair_weight=args.pair_weight,
                )
                representation = representation + float(args.rgb_mask_weight) * rgb_pair_only

            if is_normal.any() and crack_rows.any():
                xn = x[is_normal]
                donors = y[crack_rows]
                ids = torch.randint(
                    0,
                    donors.shape[0],
                    (xn.shape[0],),
                    device=device,
                    generator=corruption_generator,
                )
                donor = donors[ids]
                representation = representation + float(args.normal_critic_weight) * _critic_term(
                    critic, xn, donor, donor.clone(), args
                )

            endpoint, endpoint_terms = critic_endpoint_energy_loss(
                critic,
                x,
                y,
                wrong,
                margin=args.endpoint_margin,
                anchor_weight=args.endpoint_anchor_weight,
            )
            path, path_terms = continuous_relation_path_loss(
                critic,
                x,
                y,
                wrong,
                margin=args.path_margin,
            )
            loss = (
                representation
                + float(args.endpoint_weight) * endpoint
                + float(args.path_weight) * path
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(args.grad_clip))
            optimizer.step()

            totals.append(float(loss.detach()))
            representation_losses.append(float(representation.detach()))
            endpoint_losses.append(float(endpoint.detach()))
            path_losses.append(float(path.detach()))
            endpoint_orders.append(float(endpoint_terms["endpoint_order_fraction"]))
            path_orders.append(float(path_terms["path_order_fraction"]))
            endpoint_gaps.append(float(endpoint_terms.get("endpoint_gap", 0.0)))

        if args.normal_fraction > 0 and normal_seen <= 0:
            raise RuntimeError("normal supervision requested but critic saw zero true-normal samples")
        history.append(
            {
                "epoch": epoch,
                "loss": _mean(totals),
                "representation": _mean(representation_losses),
                "energy_endpoint": _mean(endpoint_losses),
                "energy_path": _mean(path_losses),
                "endpoint_order_fraction": _mean(endpoint_orders),
                "path_order_fraction": _mean(path_orders),
                "endpoint_gap_mean": _mean(endpoint_gaps),
                "normal_samples_seen": normal_seen,
                "rgb_shuffle_samples": rgb_seen,
                "corruption_counts": corruption_counts,
            }
        )

    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "energy_head_contract": ENERGY_HEAD_CONTRACT,
        "seed": int(cfg["seed"]),
        "critic": critic.state_dict(),
        "width": int(args.critic_width),
        "config": dict(cfg),
        "manifest_file_sha256": sha256_file(args.manifest),
        "dataset_content_sha256": args._dataset_content_sha256,
        "full_gate0_certificate_sha256": sha256_file(args.full_gate0_certificate),
        "normal_fraction": float(args.normal_fraction),
        "normal_critic_weight": float(args.normal_critic_weight),
        "training_hparams": _critic_hparams(args, cfg, determinism),
        "runtime": runtime_metadata(device, determinism),
    }
    torch.save(checkpoint, out / "critic.pt")
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic


@torch.no_grad()
def energy_qualification(critic, loader, device, margin):
    """Qualify from the complete validation trajectory population.

    Concatenating per-sample energies before summarisation is required for a
    mathematically correct global median; averaging batch medians is not a
    median and could otherwise produce a false PASS.
    """
    critic.eval()
    generator = make_generator(device, 99173)
    trajectories = []
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    for batch in loader:
        x, y, is_normal = _unpack(batch)
        x, y = x.to(device), y.to(device)
        is_normal = is_normal.to(device, dtype=torch.bool)
        wrong, _ = make_corrupted_mask(
            y, true_normal=is_normal, generator=generator, image=x
        )
        trajectories.append(
            relation_energy_trajectory(
                critic, x, y, wrong, t_values=levels
            ).detach().cpu()
        )
    if not trajectories:
        return {"energy_samples": 0, "energy_finite": False}
    all_energies = torch.cat(trajectories, dim=0)
    return summarize_energy_trajectories(
        all_energies, t_values=levels, margin=margin
    )


@torch.no_grad()
def normal_donor_energy_qualification(
    critic, crack_loader, normal_loader, device, margin, max_donors=64
):
    """Qualify C8 crack-on-normal relation energy on held-out normal images.

    Donor masks come only from crack ``val`` and use deterministic reservoir
    sampling capped on CPU to keep this diagnostic representative and memory
    bounded. No normal training rows or canonical test material are opened.
    """
    donors = []
    seen = 0
    reservoir = torch.Generator(device="cpu").manual_seed(99175)
    for batch in crack_loader:
        _, y, _ = _unpack(batch)
        crack = (y[y.flatten(1).sum(1) > 0] > 0.5).to(torch.uint8).cpu()
        for row in crack:
            seen += 1
            if len(donors) < int(max_donors):
                donors.append(row.clone())
                continue
            slot = int(torch.randint(0, seen, (1,), generator=reservoir).item())
            if slot < int(max_donors):
                donors[slot] = row.clone()
    if not donors:
        return {"energy_samples": 0, "energy_finite": False}

    donor_bank = torch.stack(donors, dim=0)
    trajectories = []
    levels = (0.0, 0.25, 0.5, 0.75, 1.0)
    offset = 0
    for batch in normal_loader:
        x, y, _ = _unpack(batch)
        if bool((y.flatten(1).sum(1) > 0).any()):
            raise RuntimeError("normal_val contains non-empty target during C8 qualification")
        x, y = x.to(device), y.to(device)
        n = int(y.shape[0])
        if n == 0:
            continue
        ids = (torch.arange(n) + offset) % int(donor_bank.shape[0])
        wrong = donor_bank[ids].to(device=device, dtype=y.dtype)
        offset += n
        trajectories.append(
            relation_energy_trajectory(
                critic, x, y, wrong, t_values=levels
            ).detach().cpu()
        )
    if not trajectories:
        return {"energy_samples": 0, "energy_finite": False}
    all_energies = torch.cat(trajectories, dim=0)
    return summarize_energy_trajectories(
        all_energies, t_values=levels, margin=margin
    )


def _validate_loaded_critic(saved, args, cfg):
    return validate_critic_checkpoint(
        saved,
        args.manifest,
        cfg,
        args.normal_fraction,
        args.normal_critic_weight,
        dataset_content_sha256_value=args._dataset_content_sha256,
        expected_hparams=_critic_hparams(args, cfg, args._determinism_mode),
        full_gate0_certificate=args.full_gate0_certificate,
    )


def qualify_critic(critic, args, cfg, device, out):
    val_loader = make_loader(
        args.manifest,
        "val",
        cfg["image_size"],
        cfg["batch_size"],
        False,
        cfg.get("num_workers", 0),
        seed=int(cfg["seed"]),
        return_is_normal=True,
    )
    normal_loader = None
    normal_split = None
    if args.normal_fraction > 0:
        if not manifest_has_split(args.manifest, "normal_val"):
            raise RuntimeError("N25 critic qualification requires held-out normal_val")
        normal_split = "normal_val"
        normal_loader = make_loader(
            args.manifest,
            "normal_val",
            cfg["image_size"],
            cfg["batch_size"],
            False,
            cfg.get("num_workers", 0),
            seed=int(cfg["seed"]),
            return_is_normal=True,
        )
    representation = critic_metrics(
        critic, val_loader, device, normal_loader=normal_loader
    )
    # ``critic_metrics`` is shared with reconstructed v2.0.4 and historically
    # labels any supplied normal loader as ``normal_train``. v2.1 deliberately
    # qualifies N25 on held-out ``normal_val``; overwrite only this provenance
    # label so the report describes the loader that was actually constructed.
    representation["normal_diagnostic_split"] = normal_split
    energy = energy_qualification(critic, val_loader, device, args.path_margin)
    failures = connected_gate_failures(representation, energy)
    normal_texture_energy = None
    normal_donor_energy = None
    if normal_loader is not None:
        # C9: texture-guided false positives generated directly on held-out normals.
        normal_texture_energy = energy_qualification(
            critic, normal_loader, device, args.path_margin
        )
        # C8: crack-shaped false positives placed on held-out normal RGB using
        # crack masks from validation as deterministic donors.
        normal_donor_energy = normal_donor_energy_qualification(
            critic, val_loader, normal_loader, device, args.path_margin
        )
        for prefix, metrics in (
            ("normal_texture_", normal_texture_energy),
            ("normal_donor_", normal_donor_energy),
        ):
            failures.extend(
                prefix + item for item in relation_energy_gate_failures(metrics)
            )
    report = {
        "classification": representation,
        "energy": energy,
        "normal_texture_energy": normal_texture_energy,
        "normal_donor_energy": normal_donor_energy,
        "normal_split": normal_split,
        "failures": failures,
        "pass": not failures,
    }
    (out / "critic_qualification_v21.json").write_text(json.dumps(report, indent=2))
    return report


def _gradient_record(name, seg, term, logits, weight):
    if term is None or float(weight) == 0.0:
        return None
    d = gradient_alignment_diagnostics(seg, term, logits)
    raw_ratio = float(d["aux_to_seg_norm_ratio"])
    return {
        "name": name,
        "weight": float(weight),
        "seg_grad_norm": float(d["seg_grad_norm"]),
        "aux_grad_norm": float(d["aux_grad_norm"]),
        "raw_aux_to_seg_grad_ratio": raw_ratio,
        "effective_aux_to_seg_grad_ratio": abs(float(weight)) * raw_ratio,
        "cosine_similarity": float(d["cosine_similarity"]),
        "finite": bool(d["finite"]),
    }


def train_student(args, cfg, device, out, critic=None):
    seed = int(cfg["seed"])
    seed_all(seed)
    loader, sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        seed,
        cfg.get("num_workers", 0),
    )
    val_loader = make_loader(
        args.manifest,
        "val",
        cfg["image_size"],
        cfg["batch_size"],
        False,
        cfg.get("num_workers", 0),
        seed=seed,
    )
    student = make_student(args.student_kind, args.student_width).to(device)
    setattr(student, "_oasis_width", int(args.student_width))
    load_student_init(student, args.student_init_checkpoint, seed)
    optimizer = _optimizer(student.parameters(), args)
    aug_generator = make_generator(device, seed + 10001)
    corruption_generator = make_generator(device, seed + 20001)
    if critic is not None:
        critic.eval()
        critic.zero_grad(set_to_none=True)
        for p in critic.parameters():
            p.requires_grad_(False)

    history = []
    best_key = None
    best_state = None
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        student.train()
        totals, segs, auxs, structs = [], [], [], []
        telemetry = []
        grad_telemetry = []
        ramp = 0.0
        gradient_sampled = False
        for batch in loader:
            x, y, is_normal = _unpack(batch)
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, aug_generator)
            logits = student(x)
            seg = segmentation_loss(logits, y)
            total = seg
            aux = logits.new_zeros(())
            structural = logits.new_zeros(())
            rc_weight = 0.0
            structural_weight = 0.0
            aux_name = None
            structural_name = None

            if epoch >= args.warmup and args.mode in {"connected", "aosk_connected"}:
                pred = logits.sigmoid()
                wrong, _ = make_corrupted_mask(
                    y, true_normal=is_normal, generator=corruption_generator, image=x
                )
                with torch.no_grad():
                    gt_out = critic(x, y)
                    corrupt_out = critic(x, wrong)
                aux, extras = oasis_rc_student_loss_v2(
                    critic(x, pred),
                    gt_out,
                    corrupt_out,
                    pred,
                    y,
                    margin=args.student_margin,
                    corrupted_rank_weight=args.corrupted_rank_weight,
                    fp_weight=args.fp_weight,
                )
                ramp = min(1.0, (epoch - args.warmup + 1) / max(1, args.ramp_epochs))
                rc_weight = float(args.lambda_oasis) * ramp
                aux_name = "rc"
                total = total + rc_weight * aux
                telemetry.append(
                    {
                        k: float(v)
                        for k, v in extras.items()
                        if torch.is_tensor(v) and v.numel() == 1
                    }
                )
            elif epoch >= args.warmup and args.mode == "adversarial":
                pred = logits.sigmoid()
                aux = adversarial_pair_student_loss(critic(x, pred))
                ramp = min(1.0, (epoch - args.warmup + 1) / max(1, args.ramp_epochs))
                rc_weight = float(args.lambda_adversarial) * ramp
                aux_name = "frozen_pair_critic"
                total = total + rc_weight * aux

            if args.mode in {"aosk", "aosk_connected"}:
                structural = oriented_consistency_loss(logits, x, y)
                structural_weight = float(args.lambda_aosk)
                structural_name = "aosk"
                total = total + structural_weight * structural
            elif args.mode == "cldice":
                structural = centerline_cldice_loss(logits, y)
                structural_weight = float(args.lambda_cldice)
                structural_name = "cldice"
                total = total + structural_weight * structural

            if not gradient_sampled:
                if aux_name is not None:
                    record = _gradient_record(aux_name, seg, aux, logits, rc_weight)
                    if record:
                        grad_telemetry.append(record)
                if structural_name is not None:
                    record = _gradient_record(
                        structural_name, seg, structural, logits, structural_weight
                    )
                    if record:
                        grad_telemetry.append(record)
                gradient_sampled = bool(grad_telemetry)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_clip))
            optimizer.step()
            if critic is not None and any(p.grad is not None for p in critic.parameters()):
                raise RuntimeError("frozen critic received gradients during student update")
            totals.append(float(total.detach()))
            segs.append(float(seg.detach()))
            auxs.append(float(aux.detach()))
            structs.append(float(structural.detach()))

        validation = select_threshold(student, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": _mean(totals),
            "seg": _mean(segs),
            "aux": _mean(auxs),
            "structural": _mean(structs),
            "aux_ramp": ramp,
            "val": validation,
            "gradient_telemetry": grad_telemetry,
        }
        if telemetry:
            for key in telemetry[0]:
                row[f"rc_{key}"] = _mean([t[key] for t in telemetry if key in t])
        history.append(row)
        key = float(validation["dice"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no student checkpoint selected")
    student.load_state_dict(best_state)
    validation = select_threshold(student, val_loader, device)
    effective = {
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "trainer_contract": TRAINER_CONTRACT,
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
        "warmup": int(args.warmup),
        "ramp_epochs": int(args.ramp_epochs),
        "student_margin": float(args.student_margin),
        "corrupted_rank_weight": float(args.corrupted_rank_weight),
        "fp_weight": float(args.fp_weight),
        "lambda_oasis": float(args.lambda_oasis),
        "lambda_aosk": float(args.lambda_aosk),
        "lambda_cldice": float(args.lambda_cldice),
        "lambda_adversarial": float(args.lambda_adversarial),
        "normal_fraction": float(args.normal_fraction),
        "aosk_variant": AOSK_VARIANT,
        "B2_semantics": "frozen-pretrained-pair-critic; not jointly-trained adversarial",
        "checkpoint_selection": "max-validation-micro-dice",
        "threshold_selection": "max-validation-micro-dice",
        "gradient_telemetry": "one representative batch per active auxiliary per epoch",
    }
    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "trainer_contract": TRAINER_CONTRACT,
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
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gate0-certificate", required=True)
    p.add_argument("--full-gate0-certificate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=sorted(ARMS | {"critic"}), required=True)
    p.add_argument("--student-init-checkpoint")
    p.add_argument("--critic-checkpoint")
    p.add_argument("--student-kind", default="mobilenetv3")
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--normal-fraction", type=float, choices=(0.0, 0.25), default=0.0)
    p.add_argument("--normal-critic-weight", type=float, default=1.0)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--ramp-epochs", type=int, default=3)
    p.add_argument("--critic-width", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--crack-dice-weight", type=float, default=1.0)
    p.add_argument("--mismatch-weight", type=float, default=1.0)
    p.add_argument("--pair-weight", type=float, default=0.25)
    p.add_argument("--rgb-mask-weight", type=float, default=1.0)
    p.add_argument("--endpoint-weight", type=float, default=1.0)
    p.add_argument("--endpoint-anchor-weight", type=float, default=0.25)
    p.add_argument("--endpoint-margin", type=float, default=0.05)
    p.add_argument("--path-weight", type=float, default=1.0)
    p.add_argument("--path-margin", type=float, default=0.02)
    p.add_argument("--student-margin", type=float, default=0.10)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0)
    p.add_argument("--fp-weight", type=float, default=1.0)
    p.add_argument("--lambda-oasis", type=float, default=0.001)
    p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--lambda-cldice", type=float, default=0.1)
    p.add_argument("--lambda-adversarial", type=float, default=0.001)
    p.add_argument(
        "--determinism-mode",
        choices=("off", "best_effort", "strict"),
        default="strict",
    )
    return p


def main():
    args = parser().parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("seed", "image_size", "batch_size", "device"):
        if key not in cfg:
            raise ValueError(f"config missing {key}")
    splits = manifest_splits(args.manifest)
    if "test" in splits or "normal_test" in splits:
        raise ValueError("v2.1 trainer refuses canonical/held-out test rows")
    cert = verify_gate0_certificate(
        args.gate0_certificate,
        args.manifest,
        int(cfg["image_size"]),
        "train" if args.normal_fraction > 0 else "none",
        args.full_gate0_certificate,
    )
    if args.normal_fraction > 0:
        if "normal_train" not in splits:
            raise ValueError("N25 requires normal_train rows")
        if "normal_val" not in splits:
            raise ValueError("N25 requires held-out normal_val rows")
    args._dataset_content_sha256 = cert["dataset_content_sha256"]
    args._determinism_mode = args.determinism_mode
    device = torch.device(cfg["device"])
    seed_all(cfg["seed"])
    configure_determinism(args.determinism_mode, device.type)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    critic = None

    if args.mode == "critic":
        critic = train_critic(args, cfg, device, out, args.determinism_mode)
    elif args.mode in {"connected", "aosk_connected", "adversarial"}:
        if not args.critic_checkpoint:
            raise ValueError(f"{args.mode} requires --critic-checkpoint")
        saved = torch.load(args.critic_checkpoint, map_location=device, weights_only=False)
        _validate_loaded_critic(saved, args, cfg)
        critic = OASISRCv2Critic(width=int(saved["width"])).to(device)
        critic.load_state_dict(saved["critic"])

    if critic is not None:
        # Re-qualify from current loaded weights on every consumer launch. Stored
        # qualification is provenance; this measurement is the live safety gate.
        report = qualify_critic(critic, args, cfg, device, out)
        if report["failures"]:
            raise RuntimeError(
                "v2.1 critic qualification failed: " + ", ".join(report["failures"])
            )
        if args.mode == "critic":
            payload = torch.load(out / "critic.pt", map_location="cpu", weights_only=False)
            payload["qualification_v21"] = report
            torch.save(payload, out / "critic.pt")
            return

    if not args.student_init_checkpoint:
        raise ValueError("student runs require --student-init-checkpoint")
    return train_student(args, cfg, device, out, critic)


if __name__ == "__main__":
    main()
