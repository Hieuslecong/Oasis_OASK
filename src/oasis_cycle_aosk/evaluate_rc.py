"""Validation/test evaluation for RGB-only deployment checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ManifestDataset
from .models import (
    BiSeNetTiny,
    DSUNetLite,
    FastSCNNLite,
    LightweightSegmenter,
    MobileNetV3SmallSegmenter,
    MultiScaleLightweightSegmenter,
)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def evaluate(model, loader, threshold, device):
    model.eval()
    tp = fp = fn = 0.0
    normal = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = (model(x).sigmoid() >= threshold).float()
        tp += float((pred * y).sum())
        fp += float((pred * (1 - y)).sum())
        fn += float(((1 - pred) * y).sum())
        for j in range(y.size(0)):
            if y[j].sum() == 0:
                normal.append(float(pred[j].sum()))
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return {
        "precision": p,
        "recall": r,
        "dice_f1": 2 * tp / (2 * tp + fp + fn + 1e-8),
        "iou": tp / (tp + fp + fn + 1e-8),
        "normal_fp_pixels_mean": float(np.mean(normal)) if normal else None,
        "normal_fp_pixels_median": float(np.median(normal)) if normal else None,
        "normal_fp_images": int(sum(v > 0 for v in normal)),
        "normal_image_count": len(normal),
        "threshold": threshold,
    }


def _build_student(kind, width):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--allow-resolution-ablation", action="store_true")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    forbidden = {"critic", "aosk", "generator", "discriminator"}.intersection(ck)
    if forbidden:
        raise ValueError(
            f"deployment checkpoint contains training-only state: {sorted(forbidden)}"
        )
    if "student" not in ck:
        raise ValueError("checkpoint does not contain student state")

    kind = ck.get("student_kind", "multiscale")
    width = int(ck.get("student_width", 16))
    trained_size = int(
        ck.get("effective_config", {}).get(
            "image_size", ck.get("config", {}).get("image_size", 128)
        )
    )
    size = trained_size if a.size is None else int(a.size)
    if size != trained_size and not a.allow_resolution_ablation:
        raise ValueError(
            f"evaluation size {size} differs from trained size {trained_size}; "
            "pass --allow-resolution-ablation only for an explicit ablation"
        )

    model = _build_student(kind, width).to(device)
    model.load_state_dict(ck["student"])
    model.eval()
    threshold = (
        a.threshold
        if a.threshold is not None
        else float(ck.get("threshold_validation", 0.5))
    )
    loader = DataLoader(
        ManifestDataset(a.manifest, a.split, size),
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    result = evaluate(model, loader, threshold, device)
    result.update(
        {
            "split": a.split,
            "checkpoint_mode": ck.get("mode"),
            "checkpoint_sha256": _sha256_file(a.checkpoint),
            "student_kind": kind,
            "image_size": size,
            "trained_image_size": trained_size,
            "device": str(device),
            "inference_contract": ck.get(
                "inference_contract", "RGB-only student"
            ),
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
