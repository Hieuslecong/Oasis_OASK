#!/usr/bin/env python3
"""Append true-normal RGB rows to a canonical crack manifest without changing val/test.

Example:

python scripts/add_normal_rgb_to_manifest.py \
  --canonical-manifest /path/to/manifest_final.jsonl \
  --normal-root /hdd1/hieulc/Oasis_AOSK/datasets/structural_defects/Walls/Non-cracked \
  --out /hdd1/hieulc/Oasis_AOSK/experiments/normal_rgb_v1/manifest_with_normal.jsonl
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_rows_hash(rows):
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def list_images(root):
    root = Path(root)
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-manifest", required=True)
    p.add_argument("--normal-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-id", default="structural_defects_walls")
    p.add_argument("--train-ratio", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must satisfy 0 < ratio < 1")

    canonical_path = Path(args.canonical_manifest)
    canonical = [
        json.loads(line)
        for line in canonical_path.read_text().splitlines()
        if line.strip()
    ]
    before_val = [r for r in canonical if r.get("split") == "val"]
    before_test = [r for r in canonical if r.get("split") == "test"]

    root = Path(args.normal_root).resolve()
    candidates = list_images(root)
    if not candidates:
        raise RuntimeError(f"no images found under {root}")

    accepted = []
    rejected = []
    for image in candidates:
        try:
            with Image.open(image) as im:
                im.verify()
            accepted.append(image)
        except Exception as exc:
            rejected.append({"image": str(image), "reason": str(exc)})

    rng = random.Random(args.seed)
    accepted = list(accepted)
    rng.shuffle(accepted)
    cut = max(1, min(len(accepted) - 1, int(round(len(accepted) * args.train_ratio))))
    train_images = accepted[:cut]
    val_images = accepted[cut:]

    def row_for(image, split):
        rel = image.relative_to(root).as_posix()
        return {
            "image": str(image),
            "mask": None,
            "split": split,
            "source_id": args.source_id,
            # Split is intentionally excluded from lineage identity.
            "lineage_id": f"{args.source_id}::{rel}",
            "is_normal": True,
        }

    normal_rows = [row_for(x, "normal_train") for x in train_images]
    normal_rows += [row_for(x, "normal_val") for x in val_images]
    merged = canonical + normal_rows

    after_val = [r for r in merged if r.get("split") == "val"]
    after_test = [r for r in merged if r.get("split") == "test"]
    if canonical_rows_hash(before_val) != canonical_rows_hash(after_val):
        raise RuntimeError("canonical val rows changed while adding normal RGB")
    if canonical_rows_hash(before_test) != canonical_rows_hash(after_test):
        raise RuntimeError("canonical test rows changed while adding normal RGB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n")

    summary = {
        "canonical_manifest": str(canonical_path.resolve()),
        "canonical_rows": len(canonical),
        "normal_root": str(root),
        "normal_candidates": len(candidates),
        "normal_accepted": len(accepted),
        "normal_rejected": rejected,
        "normal_train": len(train_images),
        "normal_val": len(val_images),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "split_counts": dict(Counter(r.get("split") for r in merged)),
        "canonical_val_hash": canonical_rows_hash(before_val),
        "canonical_test_hash": canonical_rows_hash(before_test),
        "output_file_sha256": sha256_file(out),
    }
    summary_path = out.with_suffix(out.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
