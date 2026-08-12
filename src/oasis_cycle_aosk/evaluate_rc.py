"""Validation/test evaluation for RGB-only deployment checkpoints."""
import argparse
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


@torch.no_grad()
def evaluate(model, loader, threshold):
    model.eval()
    tp = fp = fn = 0.0
    normal = []
    for x, y in loader:
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
        "normal_fp_images": int(sum(v > 0 for v in normal)),
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
    p.add_argument("--out", required=True)
    a = p.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    kind = ck.get("student_kind", "multiscale")
    width = int(ck.get("student_width", 16))
    trained_size = int(ck.get("config", {}).get("image_size", 128))
    size = trained_size if a.size is None else int(a.size)
    if size != trained_size and not a.allow_resolution_ablation:
        raise ValueError(
            f"evaluation size {size} differs from trained size {trained_size}; "
            "pass --allow-resolution-ablation only for an explicit ablation"
        )

    model = _build_student(kind, width)
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
    result = evaluate(model, loader, threshold)
    result["split"] = a.split
    result["checkpoint_mode"] = ck.get("mode")
    result["student_kind"] = kind
    result["image_size"] = size
    result["trained_image_size"] = trained_size
    result["inference_contract"] = ck.get("inference_contract", "RGB-only student")
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
