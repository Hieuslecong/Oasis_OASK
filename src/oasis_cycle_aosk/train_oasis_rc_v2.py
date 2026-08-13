"""Canonical OASIS-RC v2 trainer.

Official runs accept only a train/val manifest bound to a Gate-0 certificate.
Canonical test images/masks are never opened by this process.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from oasis_rc_v2.checkpoint import (
    CHECKPOINT_SCHEMA,
    EXPERIMENT_ID,
    IMPLEMENTATION_VERSION,
    METHOD_VERSION,
    sha256_file,
    validate_critic_checkpoint,
)
from oasis_rc_v2.corruptions import CORRUPTION_NAMES, build_targets, make_corrupted_mask
from oasis_rc_v2.critic import OASISRCv2Critic
from oasis_rc_v2.losses import (
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from oasis_rc_v2.protocol import verify_gate0_certificate
from oasis_rc_v2.qualification import critic_gate_passes
from .aosk import oriented_consistency_loss
from .data import ManifestDataset
from .models import (
    BiSeNetTiny,
    DSUNetLite,
    FastSCNNLite,
    LightweightSegmenter,
    MobileNetV3SmallSegmenter,
    MultiScaleLightweightSegmenter,
)
from .samplers import MixedBatchSampler

AOSK_VARIANT = "oriented-consistency-v1"


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(mode, device_type):
    """Configure reproducibility without making the canonical CUDA path unusable.

    ``strict`` throws on an unsupported nondeterministic CUDA operation.
    ``best_effort`` uses deterministic algorithms where available and warns rather
    than aborting when the installed PyTorch/CUDA stack lacks one. All canonical
    GPU runners use best_effort and record the exact runtime in run_metadata.json.
    """
    if mode not in {"off", "best_effort", "strict"}:
        raise ValueError(f"invalid determinism mode: {mode}")
    enabled = mode != "off"
    if device_type == "cuda" and enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False if enabled else torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = enabled
        if enabled and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
    if enabled and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(enabled, warn_only=(mode == "best_effort"))


def runtime_metadata(device, determinism_mode):
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_sha = None
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "determinism_mode": determinism_mode,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git_sha": git_sha,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device)
        meta.update(
            {
                "gpu_name": p.name,
                "gpu_total_memory": int(p.total_memory),
                "gpu_compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    return meta


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def augment(x, y, generator=None):
    def r():
        return torch.rand((), device=x.device, generator=generator)

    if r() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if r() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if r() < 0.35:
        scale = 0.90 + 0.20 * r()
        bias = (r() - 0.5) * 0.08
        x = (x * scale + bias).clamp(-1, 1)
    return x, y


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    manifest,
    split,
    size,
    batch,
    shuffle,
    num_workers=0,
    seed=1337,
    return_is_normal=False,
):
    ds = ManifestDataset(manifest, split, size, return_is_normal=return_is_normal)
    gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        generator=gen,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(num_workers > 0),
    )


def make_train_loader(manifest, size, batch, normal_fraction, seed, num_workers=0):
    crack = ManifestDataset(manifest, "train", size, return_is_normal=True)
    if float(normal_fraction) <= 0:
        gen = torch.Generator().manual_seed(int(seed))
        return (
            DataLoader(
                crack,
                batch_size=batch,
                shuffle=True,
                generator=gen,
                worker_init_fn=_seed_worker if num_workers > 0 else None,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(num_workers > 0),
            ),
            None,
        )
    normal = ManifestDataset(manifest, "normal_train", size, return_is_normal=True)
    sampler = MixedBatchSampler(len(crack), len(normal), batch, normal_fraction, seed=seed)
    return (
        DataLoader(
            ConcatDataset([crack, normal]),
            batch_sampler=sampler,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
        ),
        sampler,
    )


def manifest_splits(manifest):
    return {
        json.loads(line).get("split")
        for line in Path(manifest).read_text().splitlines()
        if line.strip()
    }


def manifest_has_split(manifest, split):
    return split in manifest_splits(manifest)


def make_student(kind, width):
    if kind == "lightweight":
        return LightweightSegmenter(width=width)
    if kind == "mobilenetv3":
        return MobileNetV3SmallSegmenter()
    if kind == "dsunet":
        return DSUNetLite(width=width)
    if kind == "fastscnn":
        return FastSCNNLite(width=width)
    if kind == "bisenet":
        return BiSeNetTiny(width=width)
    return MultiScaleLightweightSegmenter(width=width)


def type_name_for_student(student):
    mapping = {
        LightweightSegmenter: "lightweight",
        MultiScaleLightweightSegmenter: "multiscale",
        MobileNetV3SmallSegmenter: "mobilenetv3",
        DSUNetLite: "dsunet",
        FastSCNNLite: "fastscnn",
        BiSeNetTiny: "bisenet",
    }
    return mapping.get(type(student), type(student).__name__)


def load_student_init(student, checkpoint, expected_seed=None):
    if not checkpoint:
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        if saved.get("student_kind") is not None and saved["student_kind"] != type_name_for_student(student):
            raise ValueError("student init kind mismatch")
        if (
            saved.get("student_width") is not None
            and hasattr(student, "_oasis_width")
            and int(saved["student_width"]) != int(student._oasis_width)
        ):
            raise ValueError("student init width mismatch")
        if (
            expected_seed is not None
            and saved.get("seed") is not None
            and int(saved["seed"]) != int(expected_seed)
        ):
            raise ValueError(
                f"student init seed mismatch: checkpoint={saved['seed']} run={expected_seed}"
            )
        state = saved.get("student", saved)
    else:
        state = saved
    student.load_state_dict(state)


def _critic_training_hparams(args, cfg, determinism_mode):
    return {
        "lr": float(args.lr),
        "critic_epochs": int(args.critic_epochs),
        "critic_width": int(args.critic_width),
        "batch_size": int(cfg["batch_size"]),
        "crack_dice_weight": float(args.crack_dice_weight),
        "mismatch_weight": float(args.mismatch_weight),
        "pair_weight": float(args.pair_weight),
        "rgb_mask_weight": float(args.rgb_mask_weight),
        "normal_critic_weight": float(args.normal_critic_weight),
        "normal_fraction": float(args.normal_fraction),
        "determinism_mode": determinism_mode,
    }


def validate_loaded_critic(saved, args, cfg):
    expected = {
        "crack_dice_weight": float(args.crack_dice_weight),
        "mismatch_weight": float(args.mismatch_weight),
        "pair_weight": float(args.pair_weight),
        "rgb_mask_weight": float(args.rgb_mask_weight),
        "normal_critic_weight": float(args.normal_critic_weight),
        "normal_fraction": float(args.normal_fraction),
    }
    return validate_critic_checkpoint(
        saved,
        args.manifest,
        cfg,
        args.normal_fraction,
        args.normal_critic_weight,
        dataset_content_sha256_value=args._dataset_content_sha256,
        expected_hparams=expected,
    )


def _mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def threshold_sweep_metrics(model, loader, device, thresholds=None, chunk_size=16):
    """Evaluate all thresholds with one model forward per validation batch."""
    model.eval()
    if thresholds is None:
        thresholds = np.arange(0.05, 0.951, 0.01)
    thresholds = [float(t) for t in thresholds]
    t_all = torch.tensor(thresholds, device=device, dtype=torch.float32)
    n = len(thresholds)
    tp = torch.zeros(n, device=device, dtype=torch.float64)
    fp = torch.zeros(n, device=device, dtype=torch.float64)
    fn = torch.zeros(n, device=device, dtype=torch.float64)
    normal_fp_pixels = torch.zeros(n, device=device, dtype=torch.float64)
    normal_fp_images = torch.zeros(n, device=device, dtype=torch.float64)
    normal_count = 0

    for batch in loader:
        x, y = batch[:2]
        x, y = x.to(device), y.to(device)
        prob = model(x).sigmoid()
        target = y > 0.5
        normal_rows = target.flatten(1).sum(1) == 0
        normal_count += int(normal_rows.sum().item())
        for start in range(0, n, int(chunk_size)):
            stop = min(n, start + int(chunk_size))
            ts = t_all[start:stop].view(1, -1, 1, 1, 1)
            pred = prob.unsqueeze(1) >= ts
            truth = target.unsqueeze(1)
            dims = (0, 2, 3, 4)
            tp[start:stop] += (pred & truth).sum(dims, dtype=torch.float64)
            fp[start:stop] += (pred & ~truth).sum(dims, dtype=torch.float64)
            fn[start:stop] += (~pred & truth).sum(dims, dtype=torch.float64)
            if normal_rows.any():
                pn = pred[normal_rows]
                normal_fp_pixels[start:stop] += pn.sum(
                    (0, 2, 3, 4), dtype=torch.float64
                )
                normal_fp_images[start:stop] += pn.flatten(2).any(-1).sum(
                    0, dtype=torch.float64
                )

    results = []
    for i, threshold in enumerate(thresholds):
        tp_i = float(tp[i].item())
        fp_i = float(fp[i].item())
        fn_i = float(fn[i].item())
        results.append(
            {
                "precision": tp_i / (tp_i + fp_i + 1e-8),
                "recall": tp_i / (tp_i + fn_i + 1e-8),
                "dice": 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-8),
                "iou": tp_i / (tp_i + fp_i + fn_i + 1e-8),
                "normal_fp_pixels_mean": (
                    float(normal_fp_pixels[i].item()) / normal_count
                    if normal_count
                    else None
                ),
                "normal_fp_images": int(normal_fp_images[i].item()),
                "normal_image_count": normal_count,
                "threshold": threshold,
            }
        )
    return results


@torch.no_grad()
def segmentation_metrics(model, loader, device, threshold):
    return threshold_sweep_metrics(model, loader, device, [float(threshold)])[0]


@torch.no_grad()
def select_threshold(model, loader, device):
    best = None
    for metrics in threshold_sweep_metrics(model, loader, device):
        key = (
            metrics["dice"],
            -metrics["normal_fp_pixels_mean"]
            if metrics["normal_fp_pixels_mean"] is not None
            else 0.0,
        )
        if best is None or key > best[0]:
            best = (key, metrics)
    if best is None:
        raise RuntimeError("validation loader produced no batches")
    return best[1]


def _unpack_batch(batch):
    if len(batch) == 3:
        return batch
    x, y = batch
    normal = y.flatten(1).sum(1) == 0
    return x, y, normal


@torch.no_grad()
def critic_metrics(critic, loader, device, normal_loader=None):
    """Qualification diagnostics including every C1-C9 corruption and true normals."""
    gen = make_generator(device, 1729)
    critic.eval()
    crack_tp = crack_fn = 0.0
    valid_crack_predictions = 0
    rgb_good, rgb_bad, mask_good, mask_bad = [], [], [], []
    per_kind = {name: [0.0, 0.0] for name in CORRUPTION_NAMES}
    donor_bank = []

    for batch in loader:
        x, y, is_normal = _unpack_batch(batch)
        x, y = x.to(device), y.to(device)
        is_normal = is_normal.to(device, dtype=torch.bool)
        crack_rows = (~is_normal) & (y.flatten(1).sum(1) > 0)
        if not crack_rows.any():
            continue
        xc, yc = x[crack_rows], y[crack_rows]
        donor_bank.append(yc.detach().cpu())

        clean_out = critic(xc, yc)
        clean_pred = clean_out["semantic"].argmax(1)
        clean_sem, _, _ = build_targets(yc, torch.zeros_like(yc))
        crack_tp += float(((clean_pred == 1) & (clean_sem == 1)).sum())
        crack_fn += float(((clean_pred != 1) & (clean_sem == 1)).sum())
        valid_crack_predictions += int((clean_pred == 1).sum())

        rgb_good.extend(clean_out["pair"].sigmoid().flatten().cpu().tolist())
        rgb_bad.extend(critic(xc.flip(-1), yc)["pair"].sigmoid().flatten().cpu().tolist())
        flipped = yc.flip(-1)
        changed = (flipped - yc).abs().flatten(1).sum(1) > 0
        if changed.any():
            mask_good.extend(
                critic(xc[changed], yc[changed])["pair"].sigmoid().flatten().cpu().tolist()
            )
            mask_bad.extend(
                critic(xc[changed], flipped[changed])["pair"].sigmoid().flatten().cpu().tolist()
            )

        # C1-C7 and C9 are qualified on crack-positive validation rows.
        for kind in (0, 1, 2, 3, 4, 5, 6, 8):
            wrong, invalid = make_corrupted_mask(
                yc,
                true_normal=torch.zeros(yc.shape[0], device=device, dtype=torch.bool),
                generator=gen,
                forced_kinds=[kind] * yc.shape[0],
                image=xc,
            )
            sem, _, _ = build_targets(wrong, invalid)
            pred = critic(xc, wrong)["semantic"].argmax(1)
            tp = float(((pred == 2) & (sem == 2)).sum())
            fn = float(((pred != 2) & (sem == 2)).sum())
            per_kind[CORRUPTION_NAMES[kind]][0] += tp
            per_kind[CORRUPTION_NAMES[kind]][1] += fn

    donor_masks = None
    if donor_bank:
        donor_masks = torch.cat(donor_bank, 0)[:64].to(device)

    normal_bg_ok = normal_invalid = normal_pixels = 0.0
    normal_pair = []
    normal_samples = 0
    if normal_loader is not None:
        donor_cursor = 0
        for batch in normal_loader:
            xn, yn, _ = _unpack_batch(batch)
            xn, yn = xn.to(device), yn.to(device)
            clean = critic(xn, yn)
            pred = clean["semantic"].argmax(1)
            normal_bg_ok += float((pred == 0).sum())
            normal_invalid += float((pred == 2).sum())
            normal_pixels += float(pred.numel())
            normal_samples += int(xn.shape[0])
            normal_pair.extend(clean["pair"].sigmoid().flatten().cpu().tolist())

            # C8 is a real crack mask donor placed on true-normal RGB.
            if donor_masks is not None and donor_masks.shape[0] > 0:
                ids = (
                    torch.arange(xn.shape[0], device=device) + donor_cursor
                ) % donor_masks.shape[0]
                donor_cursor += xn.shape[0]
                dm = donor_masks[ids]
                sem, _, _ = build_targets(dm, dm.clone())
                cpred = critic(xn, dm)["semantic"].argmax(1)
                tp = float(((cpred == 2) & (sem == 2)).sum())
                fn = float(((cpred != 2) & (sem == 2)).sum())
                per_kind["C8_crack_on_normal"][0] += tp
                per_kind["C8_crack_on_normal"][1] += fn

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    corruption_recall = {}
    all_tp = all_fn = 0.0
    for name, (tp, fn) in per_kind.items():
        corruption_recall[name] = tp / (tp + fn + 1e-8) if tp + fn > 0 else None
        all_tp += tp
        all_fn += fn
    finite_corruption = [v for v in corruption_recall.values() if v is not None]
    rg, rb = mean_or_none(rgb_good), mean_or_none(rgb_bad)
    mg, mb = mean_or_none(mask_good), mean_or_none(mask_bad)
    return {
        "valid_crack_recall": crack_tp / (crack_tp + crack_fn + 1e-8),
        "invalid_recall": all_tp / (all_tp + all_fn + 1e-8),
        "valid_crack_predictions": valid_crack_predictions,
        "rgb_pair_drop": None if rg is None or rb is None else rg - rb,
        "mask_pair_drop": None if mg is None or mb is None else mg - mb,
        "rgb_pair_samples": len(rgb_bad),
        "mask_pair_samples": len(mask_bad),
        "valid_normal_bg_recall": (
            normal_bg_ok / normal_pixels if normal_pixels else None
        ),
        "normal_invalid_rate": (
            normal_invalid / normal_pixels if normal_pixels else None
        ),
        "normal_pair_valid_mean": mean_or_none(normal_pair),
        "normal_samples": normal_samples,
        "normal_diagnostic_split": "normal_train",
        "corruption_invalid_recall": corruption_recall,
        "min_corruption_invalid_recall": (
            min(finite_corruption) if finite_corruption else None
        ),
    }


def _critic_gate_passes(metrics):
    return critic_gate_passes(metrics)


def _critic_term(critic, x, mask, invalid, args):
    sem, mm, pv = build_targets(mask, invalid)
    return oasis_rc_critic_loss(
        critic(x, mask),
        sem,
        mm,
        pv,
        crack_dice_weight=args.crack_dice_weight,
        mismatch_weight=args.mismatch_weight,
        pair_weight=args.pair_weight,
    )[0]


def train_critic(args, cfg, device, out, determinism_mode):
    loader, sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        int(cfg["seed"]),
        cfg.get("num_workers", 0),
    )
    critic = OASISRCv2Critic(width=args.critic_width).to(device)
    opt = torch.optim.AdamW(critic.parameters(), lr=args.lr)
    aug_gen = make_generator(device, int(cfg["seed"]) + 30001)
    corrupt_gen = make_generator(device, int(cfg["seed"]) + 30002)
    history = []
    for epoch in range(args.critic_epochs):
        if sampler:
            sampler.set_epoch(epoch)
        losses = []
        counts = {}
        normals = 0
        critic.train()
        for x, y, is_normal in loader:
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, aug_gen)
            normals += int(is_normal.sum())
            wrong, invalid, meta = make_corrupted_mask(
                y,
                true_normal=is_normal,
                generator=corrupt_gen,
                return_meta=True,
                image=x,
            )
            for z in meta:
                counts[z["kind"]] = counts.get(z["kind"], 0) + 1
            loss = 0.5 * (
                _critic_term(critic, x, y, torch.zeros_like(y), args)
                + _critic_term(critic, x, wrong, invalid, args)
            )
            relational = []
            crack_rows = (~is_normal) & (y.flatten(1).sum(1) > 0)
            if crack_rows.any():
                xc, yc = x[crack_rows], y[crack_rows]
                sem, mm, pv = build_targets(yc, torch.zeros_like(yc))
                pv = torch.zeros_like(pv)
                relational.append(
                    oasis_rc_critic_loss(
                        critic(xc.flip(-1), yc),
                        sem,
                        mm,
                        pv,
                        crack_dice_weight=args.crack_dice_weight,
                        mismatch_weight=args.mismatch_weight,
                        pair_weight=args.pair_weight,
                    )[0]
                )
                mb = yc.flip(-1)
                changed = (mb - yc).abs().flatten(1).sum(1) > 0
                if changed.any():
                    relational.append(
                        _critic_term(
                            critic,
                            xc[changed],
                            mb[changed],
                            (mb[changed] - yc[changed]).abs(),
                            args,
                        )
                    )
            if relational:
                loss = loss + args.rgb_mask_weight * torch.stack(relational).mean()
            if is_normal.any() and crack_rows.any():
                xn = x[is_normal]
                cm = y[crack_rows]
                idx = torch.randint(
                    0,
                    cm.shape[0],
                    (xn.shape[0],),
                    device=y.device,
                    generator=corrupt_gen,
                )
                donor = cm[idx]
                loss = loss + args.normal_critic_weight * _critic_term(
                    critic, xn, donor, donor.clone(), args
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        if args.normal_fraction > 0 and normals <= 0:
            raise RuntimeError(
                "normal supervision requested but critic saw zero true-normal samples"
            )
        history.append(
            {
                "epoch": epoch,
                "critic_loss": _mean(losses),
                "corruption_counts": counts,
                "normal_samples_seen": normals,
            }
        )
        print(history[-1], flush=True)

    checkpoint = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "critic": critic.state_dict(),
        "width": int(args.critic_width),
        "config": dict(cfg),
        "manifest_file_sha256": sha256_file(args.manifest),
        "dataset_content_sha256": args._dataset_content_sha256,
        "gate0_certificate_sha256": sha256_file(args.gate0_certificate),
        "normal_fraction": float(args.normal_fraction),
        "normal_critic_weight": float(args.normal_critic_weight),
        "training_hparams": _critic_training_hparams(args, cfg, determinism_mode),
        "runtime": runtime_metadata(device, determinism_mode),
    }
    torch.save(checkpoint, out / "critic.pt")
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic


def train_student(args, cfg, device, out, determinism_mode, critic=None, aosk=False):
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
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)
    aug_gen = make_generator(device, seed + 10001)
    rc_gen = make_generator(device, seed + 20001)
    if critic:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)

    history = []
    best_key = None
    best_state = None
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        student.train()
        total_values = []
        seg_values = []
        rc_values = []
        aosk_values = []
        weighted_rc_values = []
        weighted_aosk_values = []
        energy_pred = []
        energy_gt = []
        energy_corrupted = []
        rc_ramp = 0.0
        for x, y, is_normal in loader:
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, aug_gen)
            logits = student(x)
            seg = segmentation_loss(logits, y)
            loss = seg
            rc_value = None
            aosk_value = None
            rc_extras = None
            if critic is not None and epoch >= args.warmup:
                pred = logits.sigmoid()
                wrong, _ = make_corrupted_mask(
                    y,
                    true_normal=is_normal,
                    generator=rc_gen,
                    image=x,
                )
                with torch.no_grad():
                    gt = critic(x, y)
                    corr = critic(x, wrong)
                rc_value, rc_extras = oasis_rc_student_loss_v2(
                    critic(x, pred),
                    gt,
                    corr,
                    pred,
                    y,
                    pair_weight=args.student_pair_weight,
                    corrupted_rank_weight=args.corrupted_rank_weight,
                )
                rc_ramp = min(
                    1.0,
                    (epoch - args.warmup + 1) / max(1, args.ramp_epochs),
                )
                loss = loss + args.lambda_oasis * rc_ramp * rc_value
            if aosk:
                aosk_value = oriented_consistency_loss(logits, x, y)
                loss = loss + args.lambda_aosk * aosk_value

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()

            total_values.append(float(loss.detach()))
            seg_values.append(float(seg.detach()))
            if rc_value is not None:
                rv = float(rc_value.detach())
                rc_values.append(rv)
                weighted_rc_values.append(float(args.lambda_oasis * rc_ramp * rv))
                energy_pred.append(float(rc_extras["e_pred"]))
                energy_gt.append(float(rc_extras["e_gt"]))
                energy_corrupted.append(float(rc_extras["e_corrupted"]))
            if aosk_value is not None:
                av = float(aosk_value.detach())
                aosk_values.append(av)
                weighted_aosk_values.append(float(args.lambda_aosk * av))

        val = select_threshold(student, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "loss_total": _mean(total_values),
            "loss_seg": _mean(seg_values),
            "loss_rc": _mean(rc_values) if rc_values else None,
            "loss_rc_weighted": _mean(weighted_rc_values) if weighted_rc_values else None,
            "loss_aosk": _mean(aosk_values) if aosk_values else None,
            "loss_aosk_weighted": _mean(weighted_aosk_values) if weighted_aosk_values else None,
            "e_pred": _mean(energy_pred) if energy_pred else None,
            "e_gt": _mean(energy_gt) if energy_gt else None,
            "e_corrupted": _mean(energy_corrupted) if energy_corrupted else None,
            "rc_ramp": rc_ramp,
            "val": val,
        }
        history.append(epoch_row)
        print(epoch_row, flush=True)
        key = (val["dice"], val["iou"])
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
        "seed": seed,
        "device": str(device),
        "image_size": int(cfg["image_size"]),
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 0)),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "mode": args.mode,
        "student_kind": args.student_kind,
        "student_width": int(args.student_width),
        "lambda_oasis": float(args.lambda_oasis),
        "lambda_aosk": float(args.lambda_aosk),
        "aosk_variant": AOSK_VARIANT if aosk else None,
        "normal_fraction": float(args.normal_fraction),
        "warmup": int(args.warmup),
        "ramp_epochs": int(args.ramp_epochs),
        "student_pair_weight": float(args.student_pair_weight),
        "corrupted_rank_weight": float(args.corrupted_rank_weight),
        "determinism_mode": determinism_mode,
    }
    runtime = runtime_metadata(device, determinism_mode)
    ck = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "student": student.state_dict(),
        "student_kind": args.student_kind,
        "student_width": int(args.student_width),
        "config": dict(cfg),
        "effective_config": effective,
        "mode": args.mode,
        "manifest_file_sha256": sha256_file(args.manifest),
        "dataset_content_sha256": args._dataset_content_sha256,
        "gate0_certificate_sha256": sha256_file(args.gate0_certificate),
        "student_init_sha256": sha256_file(args.student_init_checkpoint),
        "critic_checkpoint_sha256": sha256_file(args.critic_checkpoint),
        "threshold_validation": float(validation["threshold"]),
        "runtime": runtime,
        "inference_contract": "RGB -> crack logits only",
    }
    torch.save(ck, out / "student_only.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "validation.json").write_text(json.dumps(validation, indent=2))
    (out / "effective_config.json").write_text(json.dumps(effective, indent=2))
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_schema": CHECKPOINT_SCHEMA,
                "experiment_id": EXPERIMENT_ID,
                "method_version": METHOD_VERSION,
                "implementation_version": IMPLEMENTATION_VERSION,
                "exact_command": " ".join(shlex.quote(i) for i in sys.argv),
                "manifest_file_sha256": sha256_file(args.manifest),
                "dataset_content_sha256": args._dataset_content_sha256,
                "gate0_certificate_sha256": sha256_file(args.gate0_certificate),
                "student_init_sha256": sha256_file(args.student_init_checkpoint),
                "critic_checkpoint_sha256": sha256_file(args.critic_checkpoint),
                "effective_config": effective,
                "runtime": runtime,
                "inference_contract": "RGB -> crack logits only",
            },
            indent=2,
        )
    )
    return student, validation


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--gate0-certificate", default=None)
    p.add_argument("--allow-uncertified-manifest", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--mode",
        choices=("control", "critic", "connected", "aosk", "aosk_connected"),
        required=True,
    )
    p.add_argument("--normal-fraction", type=float, default=0.0)
    p.add_argument("--normal-critic-weight", type=float, default=1.0)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--ramp-epochs", type=int, default=3)
    p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--lambda-oasis", type=float, default=None)
    p.add_argument("--critic-width", type=int, default=None)
    p.add_argument("--crack-dice-weight", type=float, default=1.0)
    p.add_argument("--mismatch-weight", type=float, default=1.0)
    p.add_argument("--pair-weight", type=float, default=0.25)
    p.add_argument("--rgb-mask-weight", type=float, default=1.0)
    p.add_argument("--student-pair-weight", type=float, default=0.25)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0)
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--deterministic", action="store_true", help="legacy alias")
    p.add_argument(
        "--determinism-mode",
        choices=("off", "best_effort", "strict"),
        default=None,
    )
    p.add_argument("--allow-random-init", action="store_true")
    p.add_argument("--allow-inline-critic", action="store_true")
    p.add_argument(
        "--student-kind",
        choices=("multiscale", "lightweight", "mobilenetv3", "dsunet", "fastscnn", "bisenet"),
        default="multiscale",
    )
    p.add_argument("--critic-checkpoint", default=None)
    p.add_argument("--student-init-checkpoint", default=None)
    return p


def main():
    args = _build_parser().parse_args()
    if not 0 <= args.normal_fraction < 1:
        raise ValueError("--normal-fraction must satisfy 0 <= f < 1")
    if min(
        args.normal_critic_weight,
        args.crack_dice_weight,
        args.mismatch_weight,
        args.pair_weight,
        args.rgb_mask_weight,
    ) < 0:
        raise ValueError("critic loss weights must be non-negative")
    cfg = yaml.safe_load(Path(args.config).read_text())
    for key in ("seed", "image_size", "batch_size", "device"):
        if key not in cfg:
            raise ValueError(f"config missing required field: {key}")
    if args.lambda_oasis is None:
        args.lambda_oasis = float(cfg.get("lambda_oasis", 0.001))
    if args.critic_width is None:
        args.critic_width = int(cfg.get("critic_width", 8))

    splits = manifest_splits(args.manifest)
    if "test" in splits:
        raise ValueError("official trainer refuses manifests containing test rows")
    if not {"train", "val"}.issubset(splits):
        raise ValueError("training manifest must contain train and val")
    normal_policy = "train" if args.normal_fraction > 0 else "none"
    if not args.allow_uncertified_manifest:
        cert = verify_gate0_certificate(
            args.gate0_certificate,
            args.manifest,
            int(cfg["image_size"]),
            normal_policy,
        )
        args._dataset_content_sha256 = cert["dataset_content_sha256"]
    else:
        from oasis_rc_v2.protocol import dataset_content_sha256

        args._dataset_content_sha256 = dataset_content_sha256(args.manifest)

    if (
        args.mode != "critic"
        and not args.student_init_checkpoint
        and not args.allow_random_init
    ):
        raise ValueError("official student runs require --student-init-checkpoint")
    if (
        args.mode in ("connected", "aosk_connected")
        and not args.critic_checkpoint
        and not args.allow_inline_critic
    ):
        raise ValueError(
            "connected arms require one frozen --critic-checkpoint shared by S1/S3"
        )

    device_type = torch.device(cfg["device"]).type
    determinism_mode = args.determinism_mode
    if determinism_mode is None:
        determinism_mode = (
            "best_effort" if args.deterministic and device_type == "cuda"
            else "strict" if args.deterministic
            else "off"
        )
    seed_all(cfg["seed"])
    configure_determinism(determinism_mode, device_type)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA but torch.cuda.is_available() is false")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    critic = None
    if args.mode in ("critic", "connected", "aosk_connected"):
        if args.critic_checkpoint:
            saved = torch.load(
                args.critic_checkpoint, map_location=device, weights_only=False
            )
            validate_loaded_critic(saved, args, cfg)
            critic = OASISRCv2Critic(width=int(saved["width"])).to(device)
            critic.load_state_dict(saved["critic"])
            torch.save(saved, out / "critic.pt")
        else:
            critic = train_critic(args, cfg, device, out, determinism_mode)

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
        if args.normal_fraction > 0 and manifest_has_split(args.manifest, "normal_train"):
            normal_loader = make_loader(
                args.manifest,
                "normal_train",
                cfg["image_size"],
                cfg["batch_size"],
                False,
                cfg.get("num_workers", 0),
                seed=int(cfg["seed"]),
                return_is_normal=True,
            )
        metrics = critic_metrics(critic, val_loader, device, normal_loader=normal_loader)
        (out / "critic_validation.json").write_text(json.dumps(metrics, indent=2))
        print({"critic_validation": metrics}, flush=True)
        if args.mode == "critic":
            return
        if not critic_gate_passes(metrics):
            raise RuntimeError(
                "OASIS-RC v2 quality gate failed; connected training is blocked"
            )

    return train_student(
        args,
        cfg,
        device,
        out,
        determinism_mode,
        critic if args.mode in ("connected", "aosk_connected") else None,
        args.mode in ("aosk", "aosk_connected"),
    )


if __name__ == "__main__":
    main()
