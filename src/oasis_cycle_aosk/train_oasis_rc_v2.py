"""Canonical controlled training entrypoint for OASIS-RC-v2 reconstructed.

Scientific contract
-------------------
S0 control:
    L = L_seg
S1 connected:
    L = L_seg + lambda_oasis * rc_ramp * L_RCv2
S2 aosk:
    L = L_seg + lambda_aosk * L_AOSK
S3 aosk_connected:
    L = L_seg + lambda_oasis * rc_ramp * L_RCv2
              + lambda_aosk * L_AOSK

The critic and AOSK are training-only. Deployment is always RGB -> student ->
crack logits. The method-level RC-v2 formula is intentionally not redesigned in
this repair file; alternatives belong in a named v2.1/ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import ConcatDataset, DataLoader

from .aosk import oriented_consistency_loss
from .audit import audit
from .data import ManifestDataset
from .losses_v2 import (
    oasis_rc_critic_loss,
    oasis_rc_student_loss_v2,
    segmentation_loss,
)
from .models import (
    BiSeNetTiny,
    DSUNetLite,
    FastSCNNLite,
    LightweightSegmenter,
    MobileNetV3SmallSegmenter,
    MultiScaleLightweightSegmenter,
    RelationalOASISRC,
)
from .samplers import MixedBatchSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_VERSION = "2.3.2-provenance-gates"
METHOD_VERSION = "OASIS-RC-v2-reconstructed"


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def sha256_file(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_provenance():
    status = _run_git("status", "--porcelain")
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status_porcelain": status,
        "remote": _run_git("remote", "get-url", "origin"),
    }


def runtime_provenance(device):
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        result.update(
            {
                "gpu_name": torch.cuda.get_device_name(idx),
                "gpu_capability": list(torch.cuda.get_device_capability(idx)),
            }
        )
    return result


def seed_all(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_determinism(enabled):
    enabled = bool(enabled)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = enabled
        if enabled:
            torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cudnn, "allow_tf32") and enabled:
            torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        if enabled:
            torch.backends.cuda.matmul.allow_tf32 = False
    # Idempotent and fail-closed for controlled runs.
    torch.use_deterministic_algorithms(enabled, warn_only=False)


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


# ---------------------------------------------------------------------------
# Data augmentation and online corruptions
# ---------------------------------------------------------------------------

def augment(x, y, generator=None):
    """Paired image/mask augmentation driven only by its dedicated RNG."""

    def rand_scalar():
        return torch.rand((), device=x.device, generator=generator)

    if rand_scalar() < 0.5:
        x, y = x.flip(-1), y.flip(-1)
    if rand_scalar() < 0.5:
        x, y = x.flip(-2), y.flip(-2)
    if rand_scalar() < 0.35:
        scale = 0.90 + 0.20 * rand_scalar()
        bias = (rand_scalar() - 0.5) * 0.08
        x = (x * scale + bias).clamp(-1, 1)
    return x, y


def shift_zero(mask, dx=3, dy=0):
    """Zero-padded mask translation; never circular wrap-around."""
    _, _, h, w = mask.shape
    out = torch.zeros_like(mask)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    ys0, ys1 = max(0, dy), min(h, h + dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    out[..., ys0:ys1, xs0:xs1] = mask[..., src_y0:src_y1, src_x0:src_x1]
    return out


def _rand_like(x, generator=None):
    return torch.rand(
        x.shape,
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )


def make_corrupted_mask(mask, true_normal=None, generator=None):
    """Generate RC-v2 corruptions without consuming augmentation RNG.

    ``true_normal`` is explicit manifest identity. A crack-positive sample that
    happens to have an empty resized tensor is therefore not reclassified as a
    true normal. Gate 0 should block such a disappearing crack before training.
    """
    shifted = shift_zero(mask, dx=3)
    dilated = F.max_pool2d(mask, 5, 1, 2)
    eroded = -F.max_pool2d(-mask, 5, 1, 2)
    keep = (
        F.max_pool2d((_rand_like(mask, generator) > 0.985).float(), 9, 1, 4)
        < 0.5
    ).float()
    broken = mask * keep
    noise = (_rand_like(mask, generator) > 0.992).float()
    blob = (F.max_pool2d(noise, 11, 1, 5) > 0.5).float()
    donor = mask[
        torch.randperm(mask.shape[0], device=mask.device, generator=generator)
    ]

    if true_normal is None:
        normal = mask.sum((1, 2, 3)) == 0
    else:
        normal = true_normal.to(mask.device, dtype=torch.bool).view(-1)
    normal = normal.view(-1, 1, 1, 1)

    choices = torch.randint(
        0,
        5,
        (mask.shape[0], 1, 1, 1),
        device=mask.device,
        generator=generator,
    )
    candidates = torch.stack([shifted, dilated, eroded, broken, donor], dim=1)
    idx = choices.expand(-1, 1, mask.shape[-2], mask.shape[-1])
    wrong = torch.gather(candidates, 1, idx.unsqueeze(1)).squeeze(1)
    wrong = torch.where(normal, blob, wrong)
    wrong = (wrong > 0.5).float()
    invalid = (wrong - mask).abs().clamp(0, 1)
    return wrong, invalid


def build_targets(mask, invalid):
    """Build critic targets from the *actual* corruption map.

    A no-op corruption has an empty invalid map and is therefore pair-valid;
    it is never force-labeled invalid merely because a corruption function was
    called.
    """
    semantic = mask[:, 0].long()
    semantic = torch.where(
        invalid[:, 0] > 0.5,
        torch.full_like(semantic, 2),
        semantic,
    )
    mismatch = invalid
    pair_valid = (invalid.flatten(1).sum(1) == 0).float().unsqueeze(1)
    return semantic, mismatch, pair_valid


# ---------------------------------------------------------------------------
# Datasets / samplers
# ---------------------------------------------------------------------------

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
    ds = ManifestDataset(
        manifest,
        split,
        size,
        return_is_normal=return_is_normal,
    )
    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(num_workers > 0),
    )


def make_train_loader(manifest, size, batch, normal_fraction, seed, num_workers=0):
    crack_ds = ManifestDataset(
        manifest,
        "train",
        size,
        return_is_normal=True,
    )
    if float(normal_fraction) <= 0:
        generator = torch.Generator().manual_seed(int(seed))
        return (
            DataLoader(
                crack_ds,
                batch_size=batch,
                shuffle=True,
                generator=generator,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=(num_workers > 0),
            ),
            None,
        )

    normal_ds = ManifestDataset(
        manifest,
        "normal_train",
        size,
        return_is_normal=True,
    )
    joined = ConcatDataset([crack_ds, normal_ds])
    sampler = MixedBatchSampler(
        len(crack_ds),
        len(normal_ds),
        batch,
        normal_fraction,
        seed=seed,
    )
    return (
        DataLoader(
            joined,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
        ),
        sampler,
    )


def manifest_has_split(manifest, split):
    for line in Path(manifest).read_text().splitlines():
        if line.strip() and json.loads(line).get("split") == split:
            return True
    return False


# ---------------------------------------------------------------------------
# Models and checkpoint contracts
# ---------------------------------------------------------------------------

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
        saved_kind = saved.get("student_kind")
        saved_width = saved.get("student_width")
        saved_seed = saved.get("seed")
        if saved_kind is not None and saved_kind != type_name_for_student(student):
            raise ValueError(
                f"student init kind mismatch: checkpoint={saved_kind} "
                f"model={type_name_for_student(student)}"
            )
        if saved_width is not None and hasattr(student, "_oasis_width"):
            if int(saved_width) != int(student._oasis_width):
                raise ValueError(
                    f"student init width mismatch: checkpoint={saved_width} "
                    f"model={student._oasis_width}"
                )
        if expected_seed is not None and saved_seed is not None:
            if int(saved_seed) != int(expected_seed):
                raise ValueError(
                    f"student init seed mismatch: checkpoint={saved_seed} "
                    f"run={expected_seed}"
                )
        state = saved.get("student", saved)
    else:
        state = saved
    student.load_state_dict(state)


def validate_loaded_critic(saved, args, cfg):
    """Fail closed if an S1/S3 critic belongs to another controlled run."""
    required = (
        "critic",
        "config",
        "manifest_file_sha256",
        "normal_fraction",
        "normal_critic_weight",
    )
    missing = [key for key in required if key not in saved]
    if missing:
        raise ValueError(
            "critic checkpoint lacks controlled-run provenance fields: "
            + ", ".join(missing)
        )

    if saved["manifest_file_sha256"] != sha256_file(args.manifest):
        raise ValueError(
            "critic checkpoint manifest SHA256 does not match current manifest"
        )

    saved_cfg = saved.get("config", {})
    if int(saved_cfg.get("image_size", -1)) != int(cfg["image_size"]):
        raise ValueError("critic checkpoint image_size does not match current run")
    if int(saved_cfg.get("seed", -1)) != int(cfg["seed"]):
        raise ValueError("critic checkpoint seed does not match current run")

    if abs(float(saved["normal_fraction"]) - float(args.normal_fraction)) > 1e-12:
        raise ValueError(
            "critic checkpoint normal_fraction does not match current run"
        )
    if abs(
        float(saved["normal_critic_weight"]) - float(args.normal_critic_weight)
    ) > 1e-12:
        raise ValueError(
            "critic checkpoint normal_critic_weight does not match current run"
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def segmentation_metrics(model, loader, device, threshold):
    model.eval()
    tp = fp = fn = 0.0
    normal_fp = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (model(x).sigmoid() >= threshold).float()
        tp += float((pred * y).sum())
        fp += float((pred * (1 - y)).sum())
        fn += float(((1 - pred) * y).sum())
        for j in range(y.shape[0]):
            if y[j].sum() == 0:
                normal_fp.append(float(pred[j].sum()))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "dice": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "normal_fp_pixels_mean": float(np.mean(normal_fp)) if normal_fp else None,
        "normal_fp_images": int(sum(v > 0 for v in normal_fp)),
    }


@torch.no_grad()
def normal_fp_diagnostics(model, loader, device, threshold):
    model.eval()
    pixels = []
    ratios = []
    for x, y in loader:
        if float(y.sum()) != 0.0:
            raise ValueError("normal diagnostic loader contains a non-zero mask")
        x = x.to(device)
        pred = (model(x).sigmoid() >= threshold).float()
        flat = pred.flatten(1).sum(1).cpu().numpy()
        pixels.extend(float(v) for v in flat)
        ratios.extend(
            float(v) / float(pred.shape[-2] * pred.shape[-1]) for v in flat
        )
    if not pixels:
        raise RuntimeError("normal diagnostic split is empty")
    arr = np.asarray(pixels, dtype=np.float64)
    return {
        "normal_image_count": int(len(arr)),
        "threshold": float(threshold),
        "predicted_positive_pixels_mean": float(arr.mean()),
        "predicted_positive_pixels_median": float(np.median(arr)),
        "predicted_positive_ratio_mean": float(np.mean(ratios)),
        "images_with_any_fp": int((arr > 0).sum()),
        "images_with_fp_gt_10_pixels": int((arr > 10).sum()),
        "images_with_fp_gt_100_pixels": int((arr > 100).sum()),
        "max_fp_pixels": float(arr.max()),
    }


@torch.no_grad()
def select_threshold(model, loader, device):
    best = None
    for t in np.arange(0.05, 0.951, 0.01):
        metrics = segmentation_metrics(model, loader, device, float(t))
        metrics["threshold"] = float(t)
        key = (
            metrics["dice"],
            -metrics["normal_fp_pixels_mean"]
            if metrics["normal_fp_pixels_mean"] is not None
            else 0.0,
        )
        if best is None or key > best[0]:
            best = (key, metrics)
    return best[1]


@torch.no_grad()
def critic_metrics(critic, loader, device):
    """Validation-only critic qualification with no-op-safe negatives."""
    corruption_gen = make_generator(device, 1729)
    critic.eval()

    sem_correct = sem_total = 0.0
    crack_tp = crack_fn = invalid_tp = invalid_fn = 0.0
    pair_correct = pair_total = 0.0
    mismatch_scores, mismatch_labels = [], []
    valid_pair_scores = []
    rgb_valid_scores, rgb_bad_scores = [], []
    mask_valid_scores, mask_bad_scores = [], []
    rgb_tp = rgb_fn = mask_tp = mask_fn = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        crack_rows = y.flatten(1).sum(1) > 0
        wrong, invalid = make_corrupted_mask(y, generator=corruption_gen)

        for pair_kind, image, mask, inv in (
            ("valid", x, y, torch.zeros_like(y)),
            ("wrong", x, wrong, invalid),
        ):
            semantic, mismatch, pair_valid = build_targets(mask, inv)
            out = critic(image, mask)
            pred = out["semantic"].argmax(1)
            sem_correct += float((pred == semantic).sum())
            sem_total += float(semantic.numel())
            if pair_kind == "valid":
                crack_tp += float(((pred == 1) & (semantic == 1)).sum())
                crack_fn += float(((pred != 1) & (semantic == 1)).sum())
                valid_pair_scores.extend(
                    out["pair"].sigmoid().flatten().cpu().tolist()
                )
            else:
                invalid_tp += float(((pred == 2) & (semantic == 2)).sum())
                invalid_fn += float(((pred != 2) & (semantic == 2)).sum())
            pair_pred = (out["pair"].sigmoid() >= 0.5).float()
            pair_correct += float((pair_pred == pair_valid).sum())
            pair_total += float(pair_valid.numel())
            mismatch_scores.extend(
                out["mismatch"].sigmoid().flatten().cpu().tolist()
            )
            mismatch_labels.extend(mismatch.flatten().cpu().tolist())

        # RGB-flip mismatch is meaningful only for crack-positive spatial pairs.
        if crack_rows.any():
            x_c, y_c = x[crack_rows], y[crack_rows]
            valid = critic(x_c, y_c)["pair"].sigmoid()
            bad = critic(x_c.flip(-1), y_c)["pair"].sigmoid()
            rgb_valid_scores.extend(valid.flatten().cpu().tolist())
            rgb_bad_scores.extend(bad.flatten().cpu().tolist())
            rgb_tp += float((bad < 0.5).sum())
            rgb_fn += float((bad >= 0.5).sum())

            mask_bad = y_c.flip(-1)
            changed = (mask_bad - y_c).abs().flatten(1).sum(1) > 0
            if changed.any():
                x_m = x_c[changed]
                y_m = y_c[changed]
                bad_m = mask_bad[changed]
                valid_m = critic(x_m, y_m)["pair"].sigmoid()
                mask_pair = critic(x_m, bad_m)["pair"].sigmoid()
                mask_valid_scores.extend(valid_m.flatten().cpu().tolist())
                mask_bad_scores.extend(mask_pair.flatten().cpu().tolist())
                mask_tp += float((mask_pair < 0.5).sum())
                mask_fn += float((mask_pair >= 0.5).sum())

    def auc(scores, labels):
        order = np.argsort(np.asarray(scores))
        yy = np.asarray(labels)[order]
        pos = yy.sum()
        neg = len(yy) - pos
        if pos == 0 or neg == 0:
            return None
        ranks = np.arange(1, len(yy) + 1)
        return float(
            (ranks[yy == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
        )

    def mean_or_none(values):
        return float(np.mean(values)) if values else None

    rgb_valid = mean_or_none(rgb_valid_scores)
    rgb_bad = mean_or_none(rgb_bad_scores)
    mask_valid = mean_or_none(mask_valid_scores)
    mask_bad = mean_or_none(mask_bad_scores)

    return {
        "semantic_accuracy": sem_correct / (sem_total + 1e-8),
        "valid_crack_recall": crack_tp / (crack_tp + crack_fn + 1e-8),
        "invalid_recall": invalid_tp / (invalid_tp + invalid_fn + 1e-8),
        "pair_accuracy": pair_correct / (pair_total + 1e-8),
        "mismatch_auc": auc(mismatch_scores, mismatch_labels),
        "rgb_mismatch_recall": rgb_tp / (rgb_tp + rgb_fn + 1e-8),
        "mask_mismatch_recall": mask_tp / (mask_tp + mask_fn + 1e-8),
        "valid_pair_score": mean_or_none(valid_pair_scores),
        "rgb_valid_pair_score": rgb_valid,
        "rgb_pair_score": rgb_bad,
        "mask_valid_pair_score": mask_valid,
        "mask_pair_score": mask_bad,
        "rgb_pair_drop": (
            None if rgb_valid is None or rgb_bad is None else rgb_valid - rgb_bad
        ),
        "mask_pair_drop": (
            None
            if mask_valid is None or mask_bad is None
            else mask_valid - mask_bad
        ),
        "rgb_pair_samples": len(rgb_bad_scores),
        "mask_pair_samples": len(mask_bad_scores),
    }


def _critic_gate_passes(metrics):
    return all(
        (
            metrics.get("valid_crack_recall", 0.0) >= 0.80,
            metrics.get("invalid_recall", 0.0) >= 0.90,
            metrics.get("rgb_pair_drop") is not None
            and metrics["rgb_pair_drop"] >= 0.05,
            metrics.get("mask_pair_drop") is not None
            and metrics["mask_pair_drop"] >= 0.05,
            metrics.get("rgb_pair_samples", 0) > 0,
            metrics.get("mask_pair_samples", 0) > 0,
        )
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_critic(args, cfg, device, out):
    loader, mixed_sampler = make_train_loader(
        args.manifest,
        cfg["image_size"],
        cfg["batch_size"],
        args.normal_fraction,
        int(cfg["seed"]),
        cfg.get("num_workers", 0),
    )
    critic = RelationalOASISRC(width=args.critic_width).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr)

    aug_gen = make_generator(device, int(cfg["seed"]) + 30001)
    corruption_gen = make_generator(device, int(cfg["seed"]) + 30002)
    donor_gen = make_generator(device, int(cfg["seed"]) + 30003)
    history = []

    for epoch in range(args.critic_epochs):
        if mixed_sampler is not None:
            mixed_sampler.set_epoch(epoch)
        critic.train()
        epoch_losses, base_losses, normal_losses = [], [], []
        normal_samples_seen = normal_batches_seen = normal_donor_pairs = 0

        for x, y, is_normal in loader:
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, generator=aug_gen)
            crack_rows = ~is_normal

            wrong, invalid = make_corrupted_mask(
                y,
                true_normal=is_normal,
                generator=corruption_gen,
            )
            sem, mm, pv = build_targets(y, torch.zeros_like(y))
            sem_w, mm_w, pv_w = build_targets(wrong, invalid)

            base_terms = [
                oasis_rc_critic_loss(
                    critic(x, y),
                    sem,
                    mm,
                    pv,
                    pair_weight=args.pair_weight,
                ),
                oasis_rc_critic_loss(
                    critic(x, wrong),
                    sem_w,
                    mm_w,
                    pv_w,
                    pair_weight=args.pair_weight,
                ),
            ]

            if crack_rows.any():
                x_c, y_c = x[crack_rows], y[crack_rows]

                # Image/mask spatial mismatch: crack RGB is flipped but mask is not.
                sem_rgb, mm_rgb, pv_rgb = build_targets(
                    y_c,
                    torch.zeros_like(y_c),
                )
                pv_rgb = torch.zeros_like(pv_rgb)
                base_terms.append(
                    oasis_rc_critic_loss(
                        critic(x_c.flip(-1), y_c),
                        sem_rgb,
                        mm_rgb,
                        pv_rgb,
                        pair_weight=args.pair_weight,
                    )
                )

                # Mask mismatch is included only when flipping changes the mask.
                mask_bad = y_c.flip(-1)
                changed = (mask_bad - y_c).abs().flatten(1).sum(1) > 0
                if changed.any():
                    x_m = x_c[changed]
                    y_m = y_c[changed]
                    m_bad = mask_bad[changed]
                    mask_invalid = (m_bad - y_m).abs().clamp(0, 1)
                    sem_m, mm_m, pv_m = build_targets(m_bad, mask_invalid)
                    base_terms.append(
                        oasis_rc_critic_loss(
                            critic(x_m, m_bad),
                            sem_m,
                            mm_m,
                            pv_m,
                            pair_weight=args.pair_weight,
                        )
                    )

            base_loss = torch.stack(base_terms).mean()

            # True-normal relational negative: normal RGB + donor crack mask.
            normal_loss = None
            if is_normal.any():
                normal_batches_seen += 1
                normal_samples_seen += int(is_normal.sum())
            if is_normal.any() and crack_rows.any():
                x_n = x[is_normal]
                crack_masks = y[crack_rows]
                donor_idx = torch.randint(
                    0,
                    crack_masks.shape[0],
                    (x_n.shape[0],),
                    device=y.device,
                    generator=donor_gen,
                )
                donor_masks = crack_masks[donor_idx]
                sem_n, mm_n, pv_n = build_targets(donor_masks, donor_masks)
                pv_n = torch.zeros_like(pv_n)
                normal_loss = oasis_rc_critic_loss(
                    critic(x_n, donor_masks),
                    sem_n,
                    mm_n,
                    pv_n,
                    pair_weight=args.pair_weight,
                )
                normal_donor_pairs += int(x_n.shape[0])

            # Normal term has an explicit coefficient; adding it does not divide
            # or rescale the base critic objective.
            loss = base_loss
            if normal_loss is not None:
                loss = loss + args.normal_critic_weight * normal_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
            optimizer.step()

            epoch_losses.append(float(loss.detach()))
            base_losses.append(float(base_loss.detach()))
            if normal_loss is not None:
                normal_losses.append(float(normal_loss.detach()))

        row = {
            "epoch": epoch,
            "critic_loss": _mean(epoch_losses),
            "critic_base_loss": _mean(base_losses),
            "critic_normal_loss": _mean(normal_losses),
            "normal_critic_weight": float(args.normal_critic_weight),
            "requested_normal_fraction": float(args.normal_fraction),
            "realized_normal_fraction": (
                float(mixed_sampler.realized_normal_fraction)
                if mixed_sampler is not None
                else 0.0
            ),
            "optimizer_steps_per_epoch": len(loader),
            "critic_normal_samples_seen": normal_samples_seen,
            "critic_normal_batches_seen": normal_batches_seen,
            "normal_donor_negative_pairs": normal_donor_pairs,
        }
        history.append(row)
        print(row, flush=True)

    if args.normal_fraction > 0 and not any(
        row["critic_normal_samples_seen"] > 0 for row in history
    ):
        raise RuntimeError(
            "normal supervision requested but critic saw zero true-normal samples"
        )

    critic_path = out / "critic.pt"
    torch.save(
        {
            "critic": critic.state_dict(),
            "width": int(args.critic_width),
            "config": dict(cfg),
            "manifest_file_sha256": sha256_file(args.manifest),
            "normal_fraction": float(args.normal_fraction),
            "normal_critic_weight": float(args.normal_critic_weight),
        },
        critic_path,
    )
    (out / "critic_history.json").write_text(json.dumps(history, indent=2))
    return critic


def train_student(args, cfg, device, out, critic=None, aosk=False):
    seed = int(cfg["seed"])
    seed_all(seed)

    train_loader, mixed_sampler = make_train_loader(
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
    )
    normal_val_loader = None
    if manifest_has_split(args.manifest, "normal_val"):
        normal_val_loader = make_loader(
            args.manifest,
            "normal_val",
            cfg["image_size"],
            cfg["batch_size"],
            False,
            cfg.get("num_workers", 0),
        )

    student = make_student(args.student_kind, args.student_width).to(device)
    setattr(student, "_oasis_width", int(args.student_width))
    load_student_init(
        student,
        args.student_init_checkpoint,
        expected_seed=seed,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    # Dedicated RNG streams make S0/S1/S2/S3 augmentation evolution identical.
    aug_gen = make_generator(device, seed + 10001)
    rc_gen = make_generator(device, seed + 20001)

    if critic is not None:
        critic.eval()
        for parameter in critic.parameters():
            parameter.requires_grad_(False)

    history = []
    best_key = None
    best_state = None

    for epoch in range(args.epochs):
        if mixed_sampler is not None:
            mixed_sampler.set_epoch(epoch)
        student.train()
        logs = {
            name: []
            for name in (
                "loss_total",
                "loss_seg",
                "rc_total_raw",
                "rc_total_weighted",
                "rank_gt",
                "rank_corrupted",
                "rc_fp",
                "e_pred",
                "e_gt",
                "e_corrupted",
                "delta_pred_gt",
                "delta_pred_corrupted",
                "aosk_raw",
                "aosk_weighted",
            )
        }
        rc_ramp = 0.0

        for x, y, is_normal in train_loader:
            x, y = x.to(device), y.to(device)
            is_normal = is_normal.to(device, dtype=torch.bool)
            x, y = augment(x, y, generator=aug_gen)

            logits = student(x)
            seg = segmentation_loss(logits, y)
            loss = seg
            logs["loss_seg"].append(float(seg.detach()))

            if critic is not None and epoch >= args.warmup:
                pred_mask = logits.sigmoid()
                wrong_mask, _ = make_corrupted_mask(
                    y,
                    true_normal=is_normal,
                    generator=rc_gen,
                )
                with torch.no_grad():
                    gt_out = critic(x, y)
                    corrupted_out = critic(x, wrong_mask)
                pred_out = critic(x, pred_mask)
                rc, extras = oasis_rc_student_loss_v2(
                    pred_out,
                    gt_out,
                    corrupted_out,
                    pred_mask,
                    y,
                    pair_weight=args.student_pair_weight,
                    corrupted_rank_weight=args.corrupted_rank_weight,
                )
                rc_ramp = min(
                    1.0,
                    (epoch - args.warmup + 1) / max(1, args.ramp_epochs),
                )
                rc_weighted = args.lambda_oasis * rc_ramp * rc
                loss = loss + rc_weighted
                logs["rc_total_raw"].append(float(rc.detach()))
                logs["rc_total_weighted"].append(float(rc_weighted.detach()))
                key_map = {
                    "rank_gt": "rank_gt",
                    "rank_corrupted": "rank_corrupted",
                    "rc_fp": "fp",
                    "e_pred": "e_pred",
                    "e_gt": "e_gt",
                    "e_corrupted": "e_corrupted",
                    "delta_pred_gt": "delta_pred_gt",
                    "delta_pred_corrupted": "delta_pred_corrupted",
                }
                for dst, src in key_map.items():
                    logs[dst].append(float(extras[src]))

            # Canonical AOSK is independent of RC warmup/ramp.
            if aosk:
                consistency = oriented_consistency_loss(logits, x, y)
                aosk_weighted = args.lambda_aosk * consistency
                loss = loss + aosk_weighted
                logs["aosk_raw"].append(float(consistency.detach()))
                logs["aosk_weighted"].append(float(aosk_weighted.detach()))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            logs["loss_total"].append(float(loss.detach()))

        val = select_threshold(student, val_loader, device)
        row = {
            "epoch": epoch,
            "losses": {name: _mean(values) for name, values in logs.items()},
            "lambda_oasis": float(args.lambda_oasis),
            "lambda_aosk": float(args.lambda_aosk),
            "rc_ramp": float(rc_ramp),
            "requested_normal_fraction": float(args.normal_fraction),
            "realized_normal_fraction": (
                float(mixed_sampler.realized_normal_fraction)
                if mixed_sampler is not None
                else 0.0
            ),
            "optimizer_steps_per_epoch": len(train_loader),
            "val": val,
        }
        history.append(row)
        print(row, flush=True)

        key = (val["dice"], val["iou"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in student.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no student checkpoint selected")

    student.load_state_dict(best_state)
    validation = select_threshold(student, val_loader, device)

    normal_validation = None
    if normal_val_loader is not None:
        normal_validation = normal_fp_diagnostics(
            student,
            normal_val_loader,
            device,
            validation["threshold"],
        )
        (out / "normal_validation.json").write_text(
            json.dumps(normal_validation, indent=2)
        )

    init_sha = sha256_file(args.student_init_checkpoint)
    manifest_sha = sha256_file(args.manifest)
    realized_normal_fraction = (
        float(mixed_sampler.realized_normal_fraction)
        if mixed_sampler is not None
        else 0.0
    )
    effective_config = {
        "seed": seed,
        "device": str(device),
        "deterministic": bool(args.deterministic),
        "image_size": int(cfg["image_size"]),
        "batch_size": int(cfg["batch_size"]),
        "num_workers": int(cfg.get("num_workers", 0)),
        "epochs": int(args.epochs),
        "optimizer_steps_per_epoch": len(train_loader),
        "lr": float(args.lr),
        "mode": args.mode,
        "student_kind": args.student_kind,
        "student_width": int(args.student_width),
        "critic_width": int(args.critic_width),
        "normal_fraction_requested": float(args.normal_fraction),
        "normal_fraction_realized": realized_normal_fraction,
        "normal_critic_weight": float(args.normal_critic_weight),
        "lambda_oasis": float(args.lambda_oasis),
        "lambda_aosk": float(args.lambda_aosk),
        "warmup": int(args.warmup),
        "ramp_epochs": int(args.ramp_epochs),
        "pair_weight": float(args.pair_weight),
        "student_pair_weight": float(args.student_pair_weight),
        "corrupted_rank_weight": float(args.corrupted_rank_weight),
    }
    (out / "effective_config.json").write_text(
        json.dumps(effective_config, indent=2)
    )

    checkpoint = {
        "student": student.state_dict(),
        "student_kind": args.student_kind,
        "student_width": int(args.student_width),
        "config": {**cfg, "image_size": int(cfg["image_size"])},
        "effective_config": effective_config,
        "mode": args.mode,
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "lambda_oasis": float(args.lambda_oasis),
        "lambda_aosk": float(args.lambda_aosk),
        "normal_fraction": float(args.normal_fraction),
        "student_init_sha256": init_sha,
        "manifest_file_sha256": manifest_sha,
        "threshold_validation": float(validation["threshold"]),
        "inference_contract": "RGB -> crack logits only",
    }
    torch.save(checkpoint, out / "student_only.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "validation.json").write_text(json.dumps(validation, indent=2))

    effective_critic_path = args.critic_checkpoint
    if effective_critic_path is None and critic is not None:
        candidate = out / "critic.pt"
        if candidate.exists():
            effective_critic_path = str(candidate)

    metadata = {
        "method_version": METHOD_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "exact_command": " ".join(shlex.quote(item) for item in sys.argv),
        "git": git_provenance(),
        "runtime": runtime_provenance(device),
        "effective_config": effective_config,
        "student_init_checkpoint": (
            str(Path(args.student_init_checkpoint).resolve())
            if args.student_init_checkpoint
            else None
        ),
        "student_init_sha256": init_sha,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_file_sha256": manifest_sha,
        "critic_checkpoint": (
            str(Path(effective_critic_path).resolve())
            if effective_critic_path
            else None
        ),
        "critic_checkpoint_sha256": sha256_file(effective_critic_path),
        "normal_validation": normal_validation,
        "inference_contract": checkpoint["inference_contract"],
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    return student, validation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--mode",
        choices=("control", "critic", "connected", "aosk", "aosk_connected"),
        required=True,
    )
    p.add_argument("--test-split", default="test")
    p.add_argument("--normal-fraction", type=float, default=0.0)
    p.add_argument("--normal-critic-weight", type=float, default=1.0)
    p.add_argument("--critic-epochs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--ramp-epochs", type=int, default=3)
    p.add_argument("--lambda-aosk", type=float, default=0.01)
    p.add_argument("--lambda-oasis", type=float, default=None)
    p.add_argument("--critic-width", type=int, default=None)
    p.add_argument("--pair-weight", type=float, default=0.25)
    p.add_argument("--student-pair-weight", type=float, default=0.25)
    p.add_argument("--corrupted-rank-weight", type=float, default=1.0)
    p.add_argument("--student-width", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--allow-random-init", action="store_true")
    p.add_argument("--allow-inline-critic", action="store_true")
    p.add_argument(
        "--student-kind",
        choices=(
            "multiscale",
            "lightweight",
            "mobilenetv3",
            "dsunet",
            "fastscnn",
            "bisenet",
        ),
        default="multiscale",
    )
    p.add_argument("--critic-checkpoint", default=None)
    p.add_argument("--student-init-checkpoint", default=None)
    return p


def main():
    args = _build_parser().parse_args()

    if not 0.0 <= args.normal_fraction < 1.0:
        raise ValueError("--normal-fraction must satisfy 0 <= f < 1")
    if args.normal_critic_weight < 0:
        raise ValueError("--normal-critic-weight must be >= 0")
    if args.critic_epochs <= 0 or args.epochs <= 0:
        raise ValueError("epoch counts must be positive")
    if args.warmup < 0 or args.ramp_epochs <= 0:
        raise ValueError("warmup must be >=0 and ramp-epochs must be >0")

    cfg = yaml.safe_load(Path(args.config).read_text())
    for required in ("seed", "image_size", "batch_size", "device"):
        if required not in cfg:
            raise ValueError(f"config missing required field: {required}")

    if args.lambda_oasis is None:
        args.lambda_oasis = float(cfg.get("lambda_oasis", 0.001))
    if args.critic_width is None:
        args.critic_width = int(cfg.get("critic_width", 8))

    student_mode = args.mode != "critic"
    if student_mode and not args.student_init_checkpoint and not args.allow_random_init:
        raise ValueError(
            "official student runs require --student-init-checkpoint; "
            "--allow-random-init is debug-only"
        )
    if args.mode in ("connected", "aosk_connected"):
        if not args.critic_checkpoint and not args.allow_inline_critic:
            raise ValueError(
                "connected arms require one frozen --critic-checkpoint shared by "
                "S1/S3; --allow-inline-critic is debug-only"
            )

    normal_policy = "train" if args.normal_fraction > 0 else "none"
    errors = audit(
        args.manifest,
        test_split=args.test_split,
        normal_policy=normal_policy,
        resize_size=int(cfg["image_size"]),
    )
    if errors:
        raise RuntimeError("G0 FAIL:\n" + "\n".join(errors))

    seed_all(int(cfg["seed"]))
    configure_determinism(args.deterministic)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "config requests CUDA but torch.cuda.is_available() is false"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    critic = None
    use_aosk = args.mode in ("aosk", "aosk_connected")

    if args.mode in ("critic", "connected", "aosk_connected"):
        if args.critic_checkpoint:
            saved = torch.load(
                args.critic_checkpoint,
                map_location=device,
                weights_only=False,
            )
            validate_loaded_critic(saved, args, cfg)
            critic = RelationalOASISRC(
                width=int(saved.get("width", args.critic_width))
            ).to(device)
            critic.load_state_dict(saved["critic"])
            # Preserve the exact critic used by this run for audit convenience.
            torch.save(saved, out / "critic.pt")
        else:
            critic = train_critic(args, cfg, device, out)

        critic_validation = critic_metrics(
            critic,
            make_loader(
                args.manifest,
                "val",
                cfg["image_size"],
                cfg["batch_size"],
                False,
                cfg.get("num_workers", 0),
            ),
            device,
        )
        (out / "critic_validation.json").write_text(
            json.dumps(critic_validation, indent=2)
        )
        print({"critic_validation": critic_validation}, flush=True)

        if args.mode == "critic":
            (out / "run_metadata.json").write_text(
                json.dumps(
                    {
                        "method_version": METHOD_VERSION,
                        "implementation_version": IMPLEMENTATION_VERSION,
                        "exact_command": " ".join(
                            shlex.quote(item) for item in sys.argv
                        ),
                        "git": git_provenance(),
                        "runtime": runtime_provenance(device),
                        "manifest": str(Path(args.manifest).resolve()),
                        "manifest_file_sha256": sha256_file(args.manifest),
                        "critic_checkpoint": str((out / "critic.pt").resolve()),
                        "critic_checkpoint_sha256": sha256_file(out / "critic.pt"),
                        "critic_validation": critic_validation,
                    },
                    indent=2,
                )
            )
            return

        if not _critic_gate_passes(critic_validation):
            raise RuntimeError(
                "OASIS-RC quality gate failed; connected training is blocked"
            )

    train_student(
        args,
        cfg,
        device,
        out,
        critic if args.mode in ("connected", "aosk_connected") else None,
        aosk=use_aosk,
    )


if __name__ == "__main__":
    main()
