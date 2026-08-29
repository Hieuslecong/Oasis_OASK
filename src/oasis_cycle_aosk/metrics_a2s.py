"""Evaluation/calibration utilities for OASIS-A2S v0.1.3.

Evaluation is deliberately separated from training. Threshold calibration makes a
single model forward pass on CAL, caches probabilities/targets, scores the frozen
threshold grid without structural metrics, then computes crack-structure metrics
once at the selected operating point.

Per-image Dice/clDice/Boundary-F1 are defined on crack-positive samples only.
True-normal images are evaluated separately through predicted-positive fraction;
this avoids treating an empty target as an ordinary overlap sample. Paired
bootstrap inference is cluster-aware by ``lineage_id``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize

from .a2s import stage1_real_class_logits

_EPS = 1e-8


def crack_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("expected Bx2xHxW real-class logits")
    return logits.softmax(1)[:, 1:2]


def _binary_erode(mask: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1)


def _binary_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, 3, 1, 1)


def _hard_morphological_skeleton(mask: torch.Tensor) -> torch.Tensor:
    """Hard binary skeleton for evaluation-only clDice."""
    arr = mask.detach().cpu().numpy() > 0.5
    out = np.zeros_like(arr, dtype=np.float32)
    for b in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            out[b, c] = skeletonize(arr[b, c]).astype(np.float32)
    return torch.from_numpy(out)


def _binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    return (mask - _binary_erode(mask)).clamp(0, 1)


def _boundary_tolerance_for_width(width: int) -> int:
    """Resolution-scaled boundary tolerance (~0.5% of square-image diagonal)."""
    return max(1, int(round(0.005 * np.sqrt(2.0) * float(width))))


def _cldice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    if float(pred.sum()) == 0.0 and float(target.sum()) == 0.0:
        return 1.0
    if float(pred.sum()) == 0.0 or float(target.sum()) == 0.0:
        return 0.0
    skel_p = _hard_morphological_skeleton(pred)
    skel_t = _hard_morphological_skeleton(target)
    tprec = float((skel_p * target).sum()) / (float(skel_p.sum()) + _EPS)
    tsens = float((skel_t * pred).sum()) / (float(skel_t.sum()) + _EPS)
    return 2.0 * tprec * tsens / (tprec + tsens + _EPS)


def _boundary_f1(pred: torch.Tensor, target: torch.Tensor, tolerance: int) -> float:
    if int(tolerance) < 1:
        raise ValueError("boundary tolerance must be >=1 pixel")
    bp = _binary_boundary(pred)
    bt = _binary_boundary(target)
    npred, ntgt = float(bp.sum()), float(bt.sum())
    if npred == 0.0 and ntgt == 0.0:
        return 1.0
    if npred == 0.0 or ntgt == 0.0:
        return 0.0
    k = 2 * int(tolerance) + 1
    dil_t = F.max_pool2d(bt, k, 1, int(tolerance))
    dil_p = F.max_pool2d(bp, k, 1, int(tolerance))
    precision = float((bp * dil_t).sum()) / (npred + _EPS)
    recall = float((bt * dil_p).sum()) / (ntgt + _EPS)
    return 2.0 * precision * recall / (precision + recall + _EPS)


def _meta_values(meta, batch_size: int, offset: int) -> list[dict]:
    default = [
        {"sample_key": f"row-{offset+i}", "source_id": "unknown", "lineage_id": f"row-{offset+i}"}
        for i in range(batch_size)
    ]
    if meta is None:
        return default
    if isinstance(meta, dict):
        out = []
        for i in range(batch_size):
            row = {}
            for key in ("sample_key", "source_id", "lineage_id"):
                value = meta.get(key)
                if isinstance(value, (list, tuple)):
                    value = value[i]
                elif torch.is_tensor(value) and value.ndim > 0:
                    value = value[i].item()
                row[key] = str(value) if value is not None else default[i][key]
            out.append(row)
        return out
    if isinstance(meta, (list, tuple)) and len(meta) == batch_size:
        out = []
        for i, item in enumerate(meta):
            if isinstance(item, dict):
                out.append({
                    key: str(item.get(key) or default[i][key])
                    for key in ("sample_key", "source_id", "lineage_id")
                })
            else:
                out.append(default[i])
        return out
    return default


@torch.no_grad()
def collect_probability_cache(model, loader, device, *, stage1: bool = False) -> list[dict]:
    """Run the network once and cache per-image probabilities/targets on CPU."""
    model.eval()
    cache: list[dict] = []
    offset = 0
    for batch in loader:
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        x, y = x.to(device), y.to(device)
        logits = stage1_real_class_logits(model, x) if stage1 else model(x)
        probs = crack_probability(logits).detach().cpu()
        targets = y.detach().cpu()
        metas = _meta_values(meta, x.shape[0], offset)
        for i in range(x.shape[0]):
            cache.append({**metas[i], "prob": probs[i:i+1], "target": targets[i:i+1]})
        offset += x.shape[0]
    return cache


def _row_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float, meta: dict, *, structural: bool) -> dict:
    pred = (prob >= float(threshold)).float()
    target = (target > 0.5).float()
    tp = float((pred * target).sum())
    fp = float((pred * (1.0 - target)).sum())
    fn = float(((1.0 - pred) * target).sum())
    tn = float(((1.0 - pred) * (1.0 - target)).sum())
    is_normal = float(target.sum()) == 0.0

    dice = None if is_normal else 2.0 * tp / (2.0 * tp + fp + fn + _EPS)
    iou = None if is_normal else tp / (tp + fp + fn + _EPS)
    precision = None if is_normal else tp / (tp + fp + _EPS)
    recall = None if is_normal else tp / (tp + fn + _EPS)
    cldice = None
    boundary = None
    tolerance = _boundary_tolerance_for_width(int(target.shape[-1]))
    if structural and not is_normal:
        cldice = _cldice(pred, target)
        boundary = _boundary_f1(pred, target, tolerance)

    return {
        "sample_key": str(meta["sample_key"]),
        "source_id": str(meta["source_id"]),
        "lineage_id": str(meta["lineage_id"]),
        "precision": precision,
        "recall": recall,
        "dice_f1": dice,
        "iou": iou,
        "cldice": cldice,
        "boundary_f1": boundary,
        "boundary_tolerance_px": tolerance,
        "predicted_positive_fraction": float(pred.mean()),
        "is_normal": is_normal,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def records_from_cache(cache: list[dict], threshold: float, *, structural: bool) -> list[dict]:
    return [
        _row_metrics(
            item["prob"], item["target"], threshold,
            {k: item[k] for k in ("sample_key", "source_id", "lineage_id")},
            structural=structural,
        )
        for item in cache
    ]


def aggregate_records(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no evaluation rows")
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    normals = [r for r in rows if r["is_normal"]]
    cracks = [r for r in rows if not r["is_normal"]]
    crack_dice = [r["dice_f1"] for r in cracks if r.get("dice_f1") is not None]
    crack_cldice = [r["cldice"] for r in cracks if r.get("cldice") is not None]
    crack_boundary = [r["boundary_f1"] for r in cracks if r.get("boundary_f1") is not None]
    tolerances = sorted({r["boundary_tolerance_px"] for r in rows if r.get("boundary_tolerance_px")})
    mean_crack_dice = float(np.mean(crack_dice)) if crack_dice else None
    mean_crack_cldice = float(np.mean(crack_cldice)) if crack_cldice else None
    mean_crack_boundary = float(np.mean(crack_boundary)) if crack_boundary else None
    return {
        "precision": tp / (tp + fp + _EPS),
        "recall": tp / (tp + fn + _EPS),
        "dice_f1": 2.0 * tp / (2.0 * tp + fp + fn + _EPS),
        "iou": tp / (tp + fp + fn + _EPS),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + _EPS),
        "mean_image_dice": mean_crack_dice,
        "mean_crack_image_dice": mean_crack_dice,
        "mean_image_cldice": mean_crack_cldice,
        "mean_crack_image_cldice": mean_crack_cldice,
        "mean_image_boundary_f1": mean_crack_boundary,
        "mean_crack_image_boundary_f1": mean_crack_boundary,
        "normal_fp_fraction": (
            float(np.mean([r["predicted_positive_fraction"] for r in normals])) if normals else None
        ),
        "boundary_tolerance_px": tolerances[0] if len(tolerances) == 1 else tolerances,
        "num_images": len(rows),
        "num_crack_images": len(cracks),
        "num_normal_images": len(normals),
    }


def evaluate_cache(cache: list[dict], threshold: float, *, structural: bool = True) -> tuple[dict, list[dict]]:
    rows = records_from_cache(cache, threshold, structural=structural)
    metrics = aggregate_records(rows)
    metrics["threshold"] = float(threshold)
    return metrics, rows


def evaluate_model(model, loader, device, threshold: float, *, stage1: bool = False) -> tuple[dict, list[dict]]:
    cache = collect_probability_cache(model, loader, device, stage1=stage1)
    return evaluate_cache(cache, threshold, structural=True)


def calibrate_threshold(model, loader, device, grid: Iterable[float], *, stage1: bool = False, compute_structural: bool = True) -> tuple[float, dict]:
    """Select threshold on CAL using one network forward pass."""
    candidates = [float(v) for v in grid]
    if not candidates or any(v <= 0.0 or v >= 1.0 for v in candidates):
        raise ValueError("threshold grid must contain values strictly between 0 and 1")
    cache = collect_probability_cache(model, loader, device, stage1=stage1)
    scored = []
    for threshold in candidates:
        metrics, _ = evaluate_cache(cache, threshold, structural=False)
        normal_fp = metrics["normal_fp_fraction"]
        tie_fp = float(normal_fp) if normal_fp is not None else 0.0
        scored.append((metrics["dice_f1"], -tie_fp, -abs(threshold - 0.5), -threshold, threshold))
    best = max(scored)
    threshold = float(best[4])
    metrics, _ = evaluate_cache(cache, threshold, structural=bool(compute_structural))
    metrics["calibration_forward_passes"] = 1
    metrics["structural_metrics_computed"] = bool(compute_structural)
    metrics["threshold_candidates"] = len(candidates)
    return threshold, metrics


def _paired_maps(rows_a: list[dict], rows_b: list[dict], *, normal: bool) -> tuple[dict, dict, list[str]]:
    a = {r["sample_key"]: r for r in rows_a if bool(r["is_normal"]) is normal}
    b = {r["sample_key"]: r for r in rows_b if bool(r["is_normal"]) is normal}
    if set(a) != set(b):
        raise ValueError("paired evaluation rows do not have identical sample keys")
    keys = sorted(a)
    if not keys:
        raise ValueError("no paired rows for requested population")
    for key in keys:
        if a[key].get("lineage_id") != b[key].get("lineage_id"):
            raise ValueError(f"paired lineage mismatch for sample_key={key!r}")
    return a, b, keys


def _cluster_bootstrap(deltas: np.ndarray, lineages: list[str], *, seed: int, draws: int) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[str, list[float]] = defaultdict(list)
    for d, lineage in zip(deltas.tolist(), lineages):
        groups[str(lineage)].append(float(d))
    cluster_ids = sorted(groups)
    cluster_means = np.asarray([np.mean(groups[c]) for c in cluster_ids], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(cluster_means), size=(int(draws), len(cluster_means)))
    means = cluster_means[idx].mean(axis=1)
    return cluster_means, means


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], *, seed: int = 1337, draws: int = 2000) -> dict:
    """Lineage-cluster bootstrap of crack-positive per-image Dice deltas."""
    a, b, keys = _paired_maps(rows_a, rows_b, normal=False)
    delta = np.asarray([b[k]["dice_f1"] - a[k]["dice_f1"] for k in keys], dtype=np.float64)
    lineages = [str(a[k].get("lineage_id") or k) for k in keys]
    cluster_means, means = _cluster_bootstrap(delta, lineages, seed=seed, draws=draws)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "analysis_population": "crack_positive_images",
        "n": int(delta.size),
        "n_images": int(delta.size),
        "n_lineage_clusters": int(cluster_means.size),
        "mean_delta_dice": float(delta.mean()),
        "median_delta_dice": float(np.median(delta)),
        "positive_fraction": float((delta > 0).mean()),
        "cluster_mean_delta_dice": float(cluster_means.mean()),
        "positive_cluster_fraction": float((cluster_means > 0).mean()),
        "bootstrap_unit": "lineage_id",
        "bootstrap_draws": int(draws),
        "bootstrap_95ci": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def paired_normal_fp_bootstrap(rows_a: list[dict], rows_b: list[dict], *, seed: int = 1337, draws: int = 2000) -> dict | None:
    """Lineage-cluster bootstrap of normal-image predicted-positive fractions."""
    normals_a = [r for r in rows_a if r["is_normal"]]
    normals_b = [r for r in rows_b if r["is_normal"]]
    if not normals_a and not normals_b:
        return None
    a, b, keys = _paired_maps(rows_a, rows_b, normal=True)
    delta = np.asarray([
        b[k]["predicted_positive_fraction"] - a[k]["predicted_positive_fraction"] for k in keys
    ], dtype=np.float64)
    lineages = [str(a[k].get("lineage_id") or k) for k in keys]
    cluster_means, means = _cluster_bootstrap(delta, lineages, seed=seed, draws=draws)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "analysis_population": "normal_images",
        "n_images": int(delta.size),
        "n_lineage_clusters": int(cluster_means.size),
        "mean_delta_normal_fp_fraction": float(delta.mean()),
        "median_delta_normal_fp_fraction": float(np.median(delta)),
        "improved_fraction": float((delta < 0).mean()),
        "cluster_mean_delta_normal_fp_fraction": float(cluster_means.mean()),
        "bootstrap_unit": "lineage_id",
        "bootstrap_draws": int(draws),
        "bootstrap_95ci": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }
