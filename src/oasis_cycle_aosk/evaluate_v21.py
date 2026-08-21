"""Development evaluator for OASIS-RC-v2.1.

Canonical ``test`` and ``normal_test`` are both final-test material and are
intentionally unreachable here.  Pixel/topology metrics are reported on
crack-positive rows only; true-negative rows use dedicated false-positive
metrics so empty targets cannot inject structural zero Dice/IoU values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oasis_rc_v2.checkpoint import validate_student_checkpoint
from .data import ManifestDataset
from .evaluate_rc import build, manifest_splits
from .topology_ops import soft_centerline

FINAL_SPLITS = {"test", "normal_test"}


def _components(mask):
    a = np.asarray(mask, dtype=np.uint8)
    h, w = a.shape
    seen = np.zeros_like(a, dtype=bool)
    count = 0
    largest = 0
    for yy in range(h):
        for xx in range(w):
            if not a[yy, xx] or seen[yy, xx]:
                continue
            count += 1
            stack = [(yy, xx)]
            seen[yy, xx] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in (
                    (-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1),
                ):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and a[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            largest = max(largest, size)
    return count, largest


def _cldice(pred, target, iterations=10, eps=1e-6):
    p = torch.as_tensor(pred, dtype=torch.float32)[None, None]
    t = torch.as_tensor(target, dtype=torch.float32)[None, None]
    if float(t.sum()) == 0.0:
        return 1.0 if float(p.sum()) == 0.0 else 0.0, None, None
    sp = soft_centerline(p, iterations)
    st = soft_centerline(t, iterations)
    skel_precision = float((sp * t).sum() / sp.sum().clamp_min(eps))
    skel_recall = float((st * p).sum() / st.sum().clamp_min(eps))
    score = 2 * skel_precision * skel_recall / (
        skel_precision + skel_recall + eps
    )
    return float(score), skel_precision, skel_recall


def _row_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    is_normal = bool(target.sum() == 0)
    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    if is_normal:
        # Crack-overlap metrics are undefined on a true-negative row.  Keeping
        # them None prevents downstream code from averaging structural zeros.
        precision = recall = dice = iou = None
    else:
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
    cldice, skp, skr = _cldice(pred, target)
    pc, largest = _components(pred)
    tc, _ = _components(target)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "cldice": None if is_normal else cldice,
        "skeleton_precision": skp,
        "skeleton_recall": skr,
        "pred_components": pc,
        "target_components": tc,
        "component_excess": max(0, pc - tc),
        "largest_pred_component": largest,
        "fp_pixels": fp,
        "is_normal": is_normal,
        "any_fp": bool(fp > 0),
    }


@torch.no_grad()
def evaluate(model, loader, threshold, device, prediction_dir=None):
    model.eval()
    rows = []
    crack_tp = crack_fp = crack_fn = 0
    pred_dir = Path(prediction_dir) if prediction_dir else None
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    for x, y in loader:
        x = x.to(device)
        prob = model(x).sigmoid().cpu().numpy()
        truth = y.numpy()
        for j in range(len(prob)):
            pred = (prob[j, 0] >= threshold).astype(np.uint8)
            target = (truth[j, 0] > 0.5).astype(np.uint8)
            m = _row_metrics(pred, target)
            m["index"] = index
            rows.append(m)
            if not m["is_normal"]:
                target_bool = target.astype(bool)
                pred_bool = pred.astype(bool)
                crack_tp += int(np.logical_and(pred_bool, target_bool).sum())
                crack_fp += int(np.logical_and(pred_bool, ~target_bool).sum())
                crack_fn += int(np.logical_and(~pred_bool, target_bool).sum())
            if pred_dir:
                np.savez_compressed(
                    pred_dir / f"{index:06d}.npz",
                    probability=prob[j, 0].astype(np.float16),
                    prediction=pred,
                    target=target,
                )
            index += 1

    crack = [r for r in rows if not r["is_normal"]]
    normal = [r for r in rows if r["is_normal"]]

    def mean(key, subset):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    if crack:
        precision = crack_tp / (crack_tp + crack_fp + 1e-8)
        recall = crack_tp / (crack_tp + crack_fn + 1e-8)
        dice = 2 * crack_tp / (2 * crack_tp + crack_fp + crack_fn + 1e-8)
        iou = crack_tp / (crack_tp + crack_fp + crack_fn + 1e-8)
    else:
        precision = recall = dice = iou = None

    result = {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "metric_population": "crack_positive_rows_only",
        "macro_dice": mean("dice", crack),
        "macro_iou": mean("iou", crack),
        "macro_crack_dice": mean("dice", crack),
        "macro_crack_iou": mean("iou", crack),
        "cldice": mean("cldice", crack),
        "skeleton_precision": mean("skeleton_precision", crack),
        "skeleton_recall": mean("skeleton_recall", crack),
        "mean_component_excess": mean("component_excess", crack),
        "normal_fp_pixels_mean": mean("fp_pixels", normal),
        "normal_fp_components_mean": mean("pred_components", normal),
        "normal_any_fp_rate": (
            float(np.mean([r["any_fp"] for r in normal])) if normal else None
        ),
        "image_count": len(rows),
        "crack_image_count": len(crack),
        "normal_image_count": len(normal),
        "threshold": float(threshold),
        "per_image": rows,
    }
    return result


def assert_development_manifest(manifest, requested_split):
    splits = manifest_splits(manifest)
    if requested_split in FINAL_SPLITS:
        raise ValueError(
            f"canonical {requested_split} must use immutable final bundle runner"
        )
    leaked = sorted(FINAL_SPLITS.intersection(splits))
    if leaked:
        raise ValueError(
            "development evaluator refuses manifests containing final-test rows: "
            + ", ".join(leaked)
        )
    if requested_split not in splits:
        raise ValueError(f"missing split {requested_split!r}")
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    p.add_argument("--prediction-dir", default=None)
    p.add_argument("--threshold", type=float, default=None)
    a = p.parse_args()

    assert_development_manifest(a.manifest, a.split)
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    validate_student_checkpoint(ck)
    device = torch.device(a.device)
    model = build(ck["student_kind"], int(ck["student_width"])).to(device)
    model.load_state_dict(ck["student"])
    size = int(ck["effective_config"]["image_size"])
    threshold = float(
        ck["threshold_validation"] if a.threshold is None else a.threshold
    )
    loader = DataLoader(
        ManifestDataset(a.manifest, a.split, size),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    result = evaluate(model, loader, threshold, device, a.prediction_dir)
    result.update(
        {
            "split": a.split,
            "method_version": ck["method_version"],
            "implementation_version": ck["implementation_version"],
            "mode": ck["mode"],
            "inference_contract": ck["inference_contract"],
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
