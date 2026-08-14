"""Validation/final-test evaluation for RGB-only OASIS-RC v2 checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from oasis_rc_v2.checkpoint import sha256_file, validate_student_checkpoint
from oasis_rc_v2.protocol import dataset_content_sha256, verify_final_test_authorization
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
        normal.extend(
            float(pred[j].sum()) for j in range(y.size(0)) if y[j].sum() == 0
        )
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
        "threshold": float(threshold),
    }


def build(kind, width):
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


def manifest_splits(path):
    return {
        json.loads(line).get("split")
        for line in Path(path).read_text().splitlines()
        if line.strip()
    }


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
    p.add_argument("--final-test-authorization", default=None)
    a = p.parse_args()

    splits = manifest_splits(a.manifest)
    if a.split != "test" and "test" in splits:
        raise ValueError(
            "non-test evaluator refuses manifests containing canonical test rows"
        )
    if a.split not in splits:
        raise ValueError(f"requested split {a.split!r} is absent from manifest")

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    validate_student_checkpoint(ck)
    kind = ck.get("student_kind", "multiscale")
    width = int(ck.get("student_width", 16))
    trained = int(
        ck.get("effective_config", {}).get(
            "image_size", ck.get("config", {}).get("image_size", 128)
        )
    )
    size = trained if a.size is None else int(a.size)
    if size != trained and not a.allow_resolution_ablation:
        raise ValueError(f"evaluation size {size} differs from trained size {trained}")

    threshold = (
        float(a.threshold)
        if a.threshold is not None
        else float(ck.get("threshold_validation", 0.5))
    )
    authorization = None
    if a.split == "test":
        authorization = verify_final_test_authorization(
            a.final_test_authorization,
            a.checkpoint,
            a.manifest,
            threshold,
        )
        if abs(float(ck.get("threshold_validation")) - threshold) > 1e-12:
            raise ValueError(
                "canonical final-test threshold must equal the frozen validation threshold"
            )
    elif a.final_test_authorization:
        raise ValueError("final-test authorization must not be used for non-test evaluation")

    model = build(kind, width).to(device)
    model.load_state_dict(ck["student"])
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
            "checkpoint_sha256": sha256_file(a.checkpoint),
            "manifest_sha256": sha256_file(a.manifest),
            "dataset_content_sha256": dataset_content_sha256(a.manifest),
            "checkpoint_schema": ck.get("checkpoint_schema"),
            "experiment_id": ck.get("experiment_id"),
            "method_version": ck.get("method_version"),
            "implementation_version": ck.get("implementation_version"),
            "student_kind": kind,
            "image_size": size,
            "trained_image_size": trained,
            "device": str(device),
            "inference_contract": ck.get("inference_contract"),
            "final_test_authorized": authorization is not None,
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
