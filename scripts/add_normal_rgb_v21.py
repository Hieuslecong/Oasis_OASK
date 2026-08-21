#!/usr/bin/env python3
"""Append lineage-safe audited true-normal train/val/test rows for v2.1."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ACCEPTED_AUDIT_STATUSES = {"PASS", "PASS_WITH_REPAIRS"}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows_hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def _excluded_paths(path):
    if not path:
        return set()
    data = json.loads(Path(path).read_text())
    return {
        str(Path(item["path"]).resolve())
        for item in data.get("excluded_normal_candidates", [])
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-manifest", required=True)
    p.add_argument("--normal-root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lineage-regex", required=True)
    p.add_argument("--audit-summary", required=True)
    p.add_argument("--exclude-file", required=True)
    p.add_argument("--source-id", default="external_true_normal")
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    audit = json.loads(Path(a.audit_summary).read_text())
    audit_status = audit.get("status")
    if audit_status not in ACCEPTED_AUDIT_STATUSES:
        raise RuntimeError(
            "normal source technical audit must PASS or PASS_WITH_REPAIRS before v2.1 splitting"
        )
    excluded = _excluded_paths(a.exclude_file)
    if audit_status == "PASS_WITH_REPAIRS" and not excluded:
        raise RuntimeError(
            "PASS_WITH_REPAIRS requires a non-empty exclusion set; audit/exclusion provenance mismatch"
        )
    if audit_status == "PASS" and int(audit.get("derived_exclusions", 0)) > 0:
        raise RuntimeError(
            "audit reports PASS but also reports derived exclusions; provenance is inconsistent"
        )

    test_ratio = 1 - a.train_ratio - a.val_ratio
    if not (
        0 < a.train_ratio < 1
        and 0 < a.val_ratio < 1
        and 0 < test_ratio < 1
    ):
        raise ValueError("ratios must leave positive train/val/test fractions")

    canonical = [
        json.loads(x)
        for x in Path(a.canonical_manifest).read_text().splitlines()
        if x.strip()
    ]
    before_val = [r for r in canonical if r.get("split") == "val"]
    before_test = [r for r in canonical if r.get("split") == "test"]
    root = Path(a.normal_root).resolve()
    pattern = re.compile(a.lineage_regex)
    groups = defaultdict(list)
    rejected = []
    for image in sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXT
    ):
        resolved = str(image.resolve())
        if resolved in excluded:
            continue
        rel = image.relative_to(root).as_posix()
        match = pattern.search(rel)
        if not match:
            raise ValueError(f"lineage regex did not match {rel!r}")
        key = match.group(1) if match.lastindex else match.group(0)
        if not key or key.strip().lower() in {
            "unknown",
            "none",
            "n/a",
            "na",
            "placeholder",
        }:
            raise ValueError(f"invalid lineage key {key!r}")
        try:
            with Image.open(image) as im:
                im.verify()
            groups[key].append(image)
        except Exception as exc:
            rejected.append({"image": str(image), "reason": str(exc)})
    if rejected:
        raise RuntimeError("audited normal set still contains undecodable files")

    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)
    if len(keys) < 3:
        raise RuntimeError("need at least three independent normal lineage groups")
    n = len(keys)
    n_train = max(1, int(round(n * a.train_ratio)))
    n_val = max(1, int(round(n * a.val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = set(keys[:n_train])
    val = set(keys[n_train : n_train + n_val])
    test = set(keys[n_train + n_val :])
    if not test:
        raise RuntimeError("normal_test must be non-empty")
    split_by_key = {
        **{k: "normal_train" for k in train},
        **{k: "normal_val" for k in val},
        **{k: "normal_test" for k in test},
    }

    rows = []
    for key in sorted(groups):
        for image in sorted(groups[key]):
            rows.append(
                {
                    "image": str(image.resolve()),
                    "mask": None,
                    "split": split_by_key[key],
                    "source_id": a.source_id,
                    "lineage_id": f"{a.source_id}::{key}",
                    "lineage_policy": "regex-parent-v21",
                    "is_normal": True,
                    "semantic_qc_status": "audited-source-declared-normal",
                    "technical_audit_status": audit_status,
                }
            )
    merged = canonical + rows
    if _rows_hash(before_val) != _rows_hash(
        [r for r in merged if r.get("split") == "val"]
    ):
        raise RuntimeError("canonical val changed")
    if _rows_hash(before_test) != _rows_hash(
        [r for r in merged if r.get("split") == "test"]
    ):
        raise RuntimeError("canonical test changed")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError("normal lineage split overlap")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + "\n"
    )
    summary = {
        "schema": "normal-split-v21",
        "source_id": a.source_id,
        "seed": a.seed,
        "audit_status": audit_status,
        "audit_summary_sha256": _sha(a.audit_summary),
        "exclude_file_sha256": _sha(a.exclude_file),
        "lineage_groups": n,
        "train_groups": len(train),
        "val_groups": len(val),
        "test_groups": len(test),
        "normal_train": sum(r["split"] == "normal_train" for r in rows),
        "normal_val": sum(r["split"] == "normal_val" for r in rows),
        "normal_test": sum(r["split"] == "normal_test" for r in rows),
        "excluded_candidates": len(excluded),
        "split_counts": dict(Counter(r.get("split") for r in merged)),
        "canonical_val_hash": _rows_hash(before_val),
        "canonical_test_hash": _rows_hash(before_test),
        "output_sha256": _sha(out),
        "semantic_qc_status": "source-declared-normal; independent semantic certification still required for confirmatory evidence",
    }
    out.with_suffix(out.suffix + ".normal_v21.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
