#!/usr/bin/env python3
"""Audit a true-normal RGB source before adding it to training.

No model predictions are used. The script checks corrupt images, raw and decoded
RGB duplicates, cross-label duplicates against an optional cracked-reference
folder, dimensions, and writes a deterministic contact sheet for visual QC.
Exact duplicate normal RGB is a hard failure because otherwise one surface can
be over-sampled under multiple filenames.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(root):
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def decoded_rgb_sha256(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def inspect(paths, label):
    rows = []
    for path in paths:
        row = {"path": str(path), "label": label}
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                row.update(
                    {
                        "width": int(rgb.width),
                        "height": int(rgb.height),
                        "mode": str(im.mode),
                        "raw_sha256": sha256_file(path),
                        "decoded_rgb_sha256": decoded_rgb_sha256(path),
                        "status": "ok",
                    }
                )
        except Exception as exc:
            row.update({"status": "corrupt", "error": str(exc)})
        rows.append(row)
    return rows


def duplicate_groups(rows, key, labels=None):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if labels is not None and row.get("label") not in labels:
            continue
        groups[row[key]].append(row)
    return {
        digest: items
        for digest, items in groups.items()
        if len({item["path"] for item in items}) > 1
    }


def cross_label_groups(rows, key):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[row[key]].append(row)
    return {
        digest: items
        for digest, items in groups.items()
        if len({item["label"] for item in items}) > 1
    }


def write_contact_sheet(paths, out, seed, count=100, thumb=160, cols=10):
    if not paths:
        return None
    rng = random.Random(seed)
    chosen = sorted(paths)
    if len(chosen) > count:
        chosen = rng.sample(chosen, count)
    rows = (len(chosen) + cols - 1) // cols
    label_h = 24
    canvas = Image.new("RGB", (cols * thumb, rows * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(chosen):
        with Image.open(path) as im:
            tile = im.convert("RGB")
            tile.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + label_h)
        canvas.paste(tile, (x, y))
        draw.text((x + 2, y + thumb + 2), path.name[:24], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return str(out)


def serialise_groups(groups):
    return {
        digest: [
            {"path": item["path"], "label": item["label"]}
            for item in items
        ]
        for digest, items in groups.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--normal-root", required=True)
    p.add_argument("--cracked-reference-root", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--contact-sheet-count", type=int, default=100)
    args = p.parse_args()

    normal_root = Path(args.normal_root).resolve()
    cracked_root = (
        Path(args.cracked_reference_root).resolve()
        if args.cracked_reference_root
        else None
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    normal_paths = list_images(normal_root)
    if not normal_paths:
        raise RuntimeError(f"no normal images found under {normal_root}")
    cracked_paths = list_images(cracked_root) if cracked_root else []

    rows = inspect(normal_paths, "normal")
    rows += inspect(cracked_paths, "cracked_reference")

    inventory_csv = out / "inventory.csv"
    fields = [
        "path",
        "label",
        "status",
        "error",
        "width",
        "height",
        "mode",
        "raw_sha256",
        "decoded_rgb_sha256",
    ]
    with inventory_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    normal_rows = [r for r in rows if r["label"] == "normal"]
    raw_dupes = duplicate_groups(normal_rows, "raw_sha256")
    decoded_dupes = duplicate_groups(normal_rows, "decoded_rgb_sha256")
    cross_raw = cross_label_groups(rows, "raw_sha256")
    cross_decoded = cross_label_groups(rows, "decoded_rgb_sha256")
    corrupt = [r for r in rows if r.get("status") != "ok"]

    dim_counts = Counter(
        f"{r['width']}x{r['height']}"
        for r in normal_rows
        if r.get("status") == "ok"
    )
    sheet = write_contact_sheet(
        [Path(r["path"]) for r in normal_rows if r.get("status") == "ok"],
        out / "normal_contact_sheet_seed1337.jpg",
        args.seed,
        count=args.contact_sheet_count,
    )

    clean = not (
        corrupt
        or raw_dupes
        or decoded_dupes
        or cross_raw
        or cross_decoded
    )
    summary = {
        "normal_root": str(normal_root),
        "cracked_reference_root": str(cracked_root) if cracked_root else None,
        "normal_candidates": len(normal_paths),
        "cracked_reference_candidates": len(cracked_paths),
        "corrupt_count": len(corrupt),
        "normal_raw_duplicate_groups": len(raw_dupes),
        "normal_decoded_duplicate_groups": len(decoded_dupes),
        "cross_label_raw_duplicate_groups": len(cross_raw),
        "cross_label_decoded_duplicate_groups": len(cross_decoded),
        "normal_dimensions": dict(dim_counts),
        "contact_sheet": sheet,
        "status": "PASS" if clean else "FAIL",
    }
    (out / "duplicate_groups.json").write_text(
        json.dumps(
            {
                "normal_raw": serialise_groups(raw_dupes),
                "normal_decoded": serialise_groups(decoded_dupes),
                "cross_label_raw": serialise_groups(cross_raw),
                "cross_label_decoded": serialise_groups(cross_decoded),
                "corrupt": corrupt,
            },
            indent=2,
        )
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
