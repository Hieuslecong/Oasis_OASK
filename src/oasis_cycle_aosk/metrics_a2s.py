"""Evaluation/calibration utilities for OASIS-A2S v0.1.2.

These metrics are evaluation-only. The Gate-1 training loss remains unchanged so
improvements can still be attributed to OASIS pretraining/transfer rather than
to a newly introduced topology loss.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .a2s import stage1_real_class_logits
from .topology_ops import soft_centerline

_EPS = 1e-8


def crack_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("expected Bx2xHxW real-class logits")
    return logits.softmax(1)[:, 1:2]


def _binary_boundary(mask: torch.Tensor) -> torch.Tensor:
    mask = (mask > 0.5).float()
    eroded = 1.0 - F.max_pool2d(1.0 - mask, 3, 1, 1)
    return (mask - eroded).clamp(0, 1)


def _cldice(pred: torch.Tensor, target: torch.Tensor, iterations: int = 10) -> float:
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    if float(pred.sum()) == 0.0 and float(target.sum()) == 0.0:
        return 1.0
    if float(pred.sum()) == 0.0 or float(target.sum()) == 0.0:
        return 0.0
    skel_p = soft_centerline(pred, iterations)
    skel_t = soft_centerline(target, iterations)
    tprec = float((skel_p * target).sum()) / (float(skel_p.sum()) + _EPS)
    tsens = float((skel_t * pred).sum()) / (float(skel_t.sum()) + _EPS)
    return 2.0 * tprec * tsens / (tprec + tsens + _EPS)


def _boundary_f1(pred: torch.Tensor, target: torch.Tensor, tolerance: int = 1) -> float:
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


def _row_metrics(prob: torch.Tensor, target: torch.Tensor, threshold: float, key: str) -> dict:
    pred = (prob >= float(threshold)).float()
    target = (target > 0.5).float()
    tp = float((pred * target).sum())
    fp = float((pred * (1.0 - target)).sum())
    fn = float(((1.0 - pred) * target).sum())
    tn = float(((1.0 - pred) * (1.0 - target)).sum())
    dice = 2.0 * tp / (2.0 * tp + fp + fn + _EPS)
    iou = tp / (tp + fp + fn + _EPS)
    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    is_normal = float(target.sum()) == 0.0
    return {
        "sample_key": key,
        "precision": precision,
        "recall": recall,
        "dice_f1": dice,
        "iou": iou,
        "cldice": _cldice(pred, target),
        "boundary_f1": _boundary_f1(pred, target),
        "predicted_positive_fraction": float(pred.mean()),
        "is_normal": is_normal,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _meta_keys(meta, batch_size: int) -> list[str]:
    if meta is None:
        return [f"row-{i}" for i in range(batch_size)]
    if isinstance(meta, dict):
        keys = meta.get("sample_key")
        if isinstance(keys, (list, tuple)):
            return [str(v) for v in keys]
        if isinstance(keys, str):
            return [keys]
    if isinstance(meta, (list, tuple)) and len(meta) == batch_size:
        return [str(m.get("sample_key", i)) if isinstance(m, dict) else str(m) for i, m in enumerate(meta)]
    return [f"row-{i}" for i in range(batch_size)]


@torch.no_grad()
def collect_records(model, loader, device, threshold: float, *, stage1: bool = False) -> list[dict]:
    model.eval()
    rows: list[dict] = []
    offset = 0
    for batch in loader:
        if len(batch) == 3:
            x, y, meta = batch
        else:
            x, y = batch
            meta = None
        x, y = x.to(device), y.to(device)
        logits = stage1_real_class_logits(model, x) if stage1 else model(x)
        probs = crack_probability(logits)
        keys = _meta_keys(meta, x.shape[0])
        for i in range(x.shape[0]):
            key = keys[i] if keys[i] else f"row-{offset+i}"
            rows.append(_row_metrics(probs[i:i+1], y[i:i+1], threshold, key))
        offset += x.shape[0]
    return rows


def aggregate_records(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no evaluation rows")
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    tn = sum(r["tn"] for r in rows)
    normals = [r for r in rows if r["is_normal"]]
    return {
        "precision": tp / (tp + fp + _EPS),
        "recall": tp / (tp + fn + _EPS),
        "dice_f1": 2.0 * tp / (2.0 * tp + fp + fn + _EPS),
        "iou": tp / (tp + fp + fn + _EPS),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + _EPS),
        "mean_image_dice": float(np.mean([r["dice_f1"] for r in rows])),
        "mean_image_cldice": float(np.mean([r["cldice"] for r in rows])),
        "mean_image_boundary_f1": float(np.mean([r["boundary_f1"] for r in rows])),
        "normal_fp_fraction": (
            float(np.mean([r["predicted_positive_fraction"] for r in normals]))
            if normals else None
        ),
        "num_images": len(rows),
        "num_normal_images": len(normals),
    }


def evaluate_model(model, loader, device, threshold: float, *, stage1: bool = False) -> tuple[dict, list[dict]]:
    rows = collect_records(model, loader, device, threshold, stage1=stage1)
    metrics = aggregate_records(rows)
    metrics["threshold"] = float(threshold)
    return metrics, rows


def calibrate_threshold(model, loader, device, grid: Iterable[float], *, stage1: bool = False) -> tuple[float, dict]:
    candidates = [float(v) for v in grid]
    if not candidates or any(v <= 0.0 or v >= 1.0 for v in candidates):
        raise ValueError("threshold grid must contain values strictly between 0 and 1")
    scored = []
    for threshold in candidates:
        metrics, _ = evaluate_model(model, loader, device, threshold, stage1=stage1)
        normal_fp = metrics["normal_fp_fraction"]
        tie_fp = float(normal_fp) if normal_fp is not None else 0.0
        scored.append((metrics["dice_f1"], -tie_fp, -abs(threshold - 0.5), -threshold, threshold, metrics))
    best = max(scored)
    return float(best[4]), best[5]


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], *, seed: int = 1337, draws: int = 2000) -> dict:
    a = {r["sample_key"]: r for r in rows_a}
    b = {r["sample_key"]: r for r in rows_b}
    if set(a) != set(b):
        raise ValueError("paired evaluation rows do not have identical sample keys")
    keys = sorted(a)
    delta = np.asarray([b[k]["dice_f1"] - a[k]["dice_f1"] for k in keys], dtype=np.float64)
    if delta.size == 0:
        raise ValueError("no paired rows")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, delta.size, size=(int(draws), delta.size))
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "n": int(delta.size),
        "mean_delta_dice": float(delta.mean()),
        "median_delta_dice": float(np.median(delta)),
        "positive_fraction": float((delta > 0).mean()),
        "bootstrap_draws": int(draws),
        "bootstrap_95ci": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }
