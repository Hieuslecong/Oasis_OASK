#!/usr/bin/env python3
"""Append true-normal RGB rows without changing canonical val/test.

Parent/session identity must be used when an auxiliary normal validation split is
created. By default all external normals stay in ``normal_train``. To create a
normal_val split, provide ``--lineage-regex`` with a capturing group that maps
patch filenames/relative paths back to their strongest available parent.

Example (pattern depends on the actual archive naming convention):

python scripts/add_normal_rgb_to_manifest.py \
  --canonical-manifest /path/to/manifest_final.jsonl \
  --normal-root /path/to/Walls/Non-cracked \
  --lineage-regex '^(parent_[^_]+)' \
  --train-ratio 0.90 \
  --out /path/to/manifest_with_normal.jsonl
"""
import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
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
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def lineage_key(image, root, pattern, allow_file_level):
    rel = image.relative_to(root).as_posix()
    if pattern is None:
        if not allow_file_level:
            return None
        return rel
    match = pattern.search(rel)
    if not match:
        raise ValueError(
            f"lineage regex did not match {rel!r}; refuse partial lineage coverage"
        )
    if match.lastindex:
        key = match.group(1)
    else:
        key = match.group(0)
    if not key:
        raise ValueError(f"empty lineage key for {rel!r}")
    return key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-manifest", required=True)
    p.add_argument("--normal-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source-id", default="structural_defects_walls")
    p.add_argument(
        "--train-ratio",
        type=float,
        default=1.0,
        help="fraction of lineage groups assigned to normal_train; default 1.0",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--lineage-regex",
        default=None,
        help="regex over relative path; first capturing group is parent lineage",
    )
    p.add_argument(
        "--allow-file-level-lineage",
        action="store_true",
        help="debug-only escape hatch; each file becomes its own lineage",
    )
    args = p.parse_args()

    if not 0.0 < args.train_ratio <= 1.0:
        raise ValueError("--train-ratio must satisfy 0 < ratio <= 1")
    if args.train_ratio < 1.0 and not args.lineage_regex:
        raise ValueError(
            "creating normal_val requires --lineage-regex so derived patches from "
            "the same parent cannot cross normal_train/normal_val. Use "
            "--train-ratio 1.0 when parent identity is unavailable."
        )
    if args.lineage_regex and args.allow_file_level_lineage:
        raise ValueError("choose --lineage-regex or --allow-file-level-lineage, not both")

    pattern = re.compile(args.lineage_regex) if args.lineage_regex else None
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

    # If there is no parent identity and no explicit file-level escape hatch,
    # keep all normals in train. This is safe and makes the limitation explicit.
    if pattern is None and not args.allow_file_level_lineage:
        groups = {"UNRESOLVED_ALL_NORMALS": list(accepted)}
        lineage_policy = "unresolved-all-train"
    else:
        groups = defaultdict(list)
        for image in accepted:
            key = lineage_key(image, root, pattern, args.allow_file_level_lineage)
            groups[key].append(image)
        groups = dict(groups)
        lineage_policy = "regex-parent" if pattern is not None else "file-level-debug"

    group_keys = sorted(groups)
    rng = random.Random(args.seed)
    rng.shuffle(group_keys)

    if args.train_ratio >= 1.0:
        train_groups = set(group_keys)
        val_groups = set()
    else:
        if len(group_keys) < 2:
            raise RuntimeError("need at least two independent lineage groups for normal_val")
        cut = max(1, min(len(group_keys) - 1, int(round(len(group_keys) * args.train_ratio))))
        train_groups = set(group_keys[:cut])
        val_groups = set(group_keys[cut:])

    rows = []
    train_images = []
    val_images = []
    for key in sorted(groups):
        split = "normal_train" if key in train_groups else "normal_val"
        for image in sorted(groups[key]):
            rel = image.relative_to(root).as_posix()
            rows.append(
                {
                    "image": str(image),
                    "mask": None,
                    "split": split,
                    "source_id": args.source_id,
                    "lineage_id": f"{args.source_id}::{key}",
                    "lineage_policy": lineage_policy,
                    "is_normal": True,
                }
            )
            (train_images if split == "normal_train" else val_images).append(image)

    merged = canonical + rows
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
        "lineage_policy": lineage_policy,
        "lineage_regex": args.lineage_regex,
        "lineage_groups": len(groups),
        "normal_train_groups": len(train_groups),
        "normal_val_groups": len(val_groups),
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
