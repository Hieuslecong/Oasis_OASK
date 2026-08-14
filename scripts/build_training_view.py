#!/usr/bin/env python3
"""Create a canonical train/validation-only manifest consumed by official trainers."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--exclude-normal",
        action="store_true",
        help="build the N0 crack-only train/val view",
    )
    a = p.parse_args()
    rows = [
        json.loads(line)
        for line in Path(a.input).read_text().splitlines()
        if line.strip()
    ]
    allowed = {"train", "val"} if a.exclude_normal else {
        "train",
        "val",
        "normal_train",
        "normal_val",
    }
    kept = [row for row in rows if row.get("split") in allowed]
    splits = {row.get("split") for row in kept}
    if not {"train", "val"}.issubset(splits):
        raise RuntimeError("training view must contain train and val")
    if "test" in splits:
        raise RuntimeError("training view unexpectedly contains test")
    if a.exclude_normal and any(str(s).startswith("normal_") for s in splits):
        raise RuntimeError("N0 training view unexpectedly contains normal rows")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n"
    )
    print(
        json.dumps(
            {
                "input": str(Path(a.input).resolve()),
                "output": str(out.resolve()),
                "rows": len(kept),
                "splits": sorted(splits),
                "normal_included": not a.exclude_normal,
                "sha256": sha(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
