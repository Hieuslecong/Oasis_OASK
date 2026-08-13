#!/usr/bin/env python3
"""Deterministic leakage repair for the canonical OmniCrack30k crack manifest.

Implements the B+ scientific constraints:
  1. Strip benign split-qualified lineage prefix (train:BCL_c1.png -> BCL_c1.png).
  2. Re-derive leakage groups using the EXACT same hashing as oasis_cycle_aosk.audit
     (imported, never reimplemented) so Gate 0 re-run is consistent.
  3. Collapse into unique groups; report finding/unique-group/unique-row counts.
  4. Exact decoded-RGB / image-mask pair duplicates across splits = P0 leakage.
     Priority test > val > train: keep test, drop train. val<->test => BLOCK.
  5. Mask-only collisions with DIFFERENT rgb = class C (review, keep, report as
     repeated_label_geometry). Never dropped solely to satisfy Gate 0.
  6. Never touches the published dataset; only emits a derived certified manifest.

Outputs:
  manifest_clean.jsonl
  manifest_clean_report.json
  manifest_clean_decisions.csv
  manifest_clean.sha256
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# Reuse the audit module's exact hashing so the cleaner and Gate 0 never diverge.
from oasis_cycle_aosk.audit import (  # noqa: E402
    _sha256_file,
    _decoded_rgb_sha256,
    _decoded_binary_mask,
    _array_sha256,
    _pair_sha256,
    audit,
)

PRIORITY = {"test": 3, "val": 2, "train": 1}
SPLITS = ("train", "val", "test")

DECISIONS = [
    "FIX_LINEAGE_FORMAT",
    "KEEP",
    "DROP_TRAIN_EXACT_RGB_DUPLICATE",
    "DROP_TRAIN_EXACT_PAIR_DUPLICATE",
    "REVIEW_MASK_ONLY_DUPLICATE",
    "BLOCK_VAL_TEST_DUPLICATE",
]


def strip_lineage(lineage: str) -> str:
    for split in SPLITS:
        for sep in (":", "::"):
            if lineage.startswith(split + sep):
                return lineage[len(split + sep):]
    return lineage


def build_row_digests(rows):
    out = []
    for r in rows:
        d = {"decoded_rgb": None, "pair": None}
        try:
            d["decoded_rgb"] = _decoded_rgb_sha256(r["image"])
        except Exception:
            pass
        if not r.get("is_normal") and r.get("mask"):
            try:
                bm = _decoded_binary_mask(r["mask"])
                d["pair"] = _pair_sha256(d["decoded_rgb"], _array_sha256(bm, "binary-mask"))
            except Exception:
                pass
        out.append(d)
    return out


def run(input_path, out_dir, resize_size=256):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(input_path) if l.strip()]

    # ---- Step 1: fix lineage format (no split membership change) ----
    decisions = []
    for i, r in enumerate(rows):
        before = r["lineage_id"]
        after = strip_lineage(before)
        r["lineage_id"] = after
        rec = {
            "row_id": i, "image": r["image"], "mask": r.get("mask"),
            "original_split": r["split"], "lineage_id_before": before,
            "lineage_id_after": after, "issue_type": "", "duplicate_group_id": "",
            "conflicting_split": "", "decision": "KEEP", "reason": "",
            "retained_counterpart": "",
        }
        if after != before:
            rec["decision"] = "FIX_LINEAGE_FORMAT"
            rec["issue_type"] = "split_qualified_lineage"
            rec["reason"] = "stripped split prefix from lineage_id"
        decisions.append(rec)

    dig = build_row_digests(rows)

    rgb_groups = defaultdict(list)
    pair_groups = defaultdict(list)
    raw_mask_groups = defaultdict(list)
    bin_mask_groups = defaultdict(list)
    for i, r in enumerate(rows):
        if dig[i]["decoded_rgb"]:
            rgb_groups[dig[i]["decoded_rgb"]].append((i, r["split"], r["image"]))
        if dig[i]["pair"]:
            pair_groups[dig[i]["pair"]].append((i, r["split"], r["image"]))
        if not r.get("is_normal") and r.get("mask"):
            try:
                raw_mask_groups[_sha256_file(r["mask"])].append((i, r["split"], r["mask"]))
                bm = _decoded_binary_mask(r["mask"])
                bin_mask_groups[_array_sha256(bm, "binary-mask")].append((i, r["split"], r["mask"]))
            except Exception:
                pass

    def multi_split(items):
        return {s for (_, s, _) in items}

    drop_set = set()
    retained_of = {}
    repeated_geometry = []
    block_groups = []
    finding_count = 0
    unique_group_count = 0
    unique_affected_rows = set()

    # ---- Step 4: decode-RGB / pair across splits = P0 ----
    for digest, items in list(rgb_groups.items()) + list(pair_groups.items()):
        paths = {p for (_, _, p) in items}
        splits = multi_split(items)
        if len(paths) <= 1 or len(splits) <= 1:
            continue
        finding_count += 1
        unique_group_count += 1
        unique_affected_rows.update(i for (i, _, _) in items)
        is_pair = digest in pair_groups
        if {"val", "test"}.issubset(splits):
            block_groups.append(("rgb/pair", digest, sorted(splits)))
            for (i, s, p) in items:
                decisions[i].update({
                    "decision": "BLOCK_VAL_TEST_DUPLICATE",
                    "issue_type": "exact_rgb_val_test_duplicate",
                    "duplicate_group_id": f"rgb:{digest[:12]}",
                    "conflicting_split": ",".join(sorted(splits)),
                    "reason": "val<->test exact-RGB duplicate; manual resolution required",
                })
            continue
        keep_split = max(splits, key=lambda s: PRIORITY[s])
        dec = "DROP_TRAIN_EXACT_PAIR_DUPLICATE" if is_pair else "DROP_TRAIN_EXACT_RGB_DUPLICATE"
        kept_path = next(p for (i, s, p) in items if s == keep_split)
        for (i, s, p) in items:
            if s != keep_split:
                drop_set.add(i)
                retained_of[i] = kept_path
                decisions[i].update({
                    "decision": dec,
                    "issue_type": "exact_rgb_cross_split_duplicate",
                    "duplicate_group_id": f"rgb:{digest[:12]}",
                    "conflicting_split": ",".join(sorted(splits)),
                    "reason": f"exact decoded-RGB duplicate; kept {keep_split} copy",
                    "retained_counterpart": kept_path,
                })

    # ---- Step 5: mask-only collisions with DIFFERENT rgb = class C ----
    for label, groups in (("raw-mask", raw_mask_groups), ("bin-mask", bin_mask_groups)):
        for digest, items in groups.items():
            splits = multi_split(items)
            if len(splits) <= 1:
                continue
            rgbs = {dig[i]["decoded_rgb"] for (i, _, _) in items if dig[i]["decoded_rgb"]}
            if len(rgbs) <= 1:
                continue  # same rgb already handled as rgb/pair dup
            finding_count += 1
            unique_group_count += 1
            unique_affected_rows.update(i for (i, _, _) in items)
            repeated_geometry.append({
                "kind": label, "digest": digest, "splits": sorted(splits),
                "rows": [{"row_id": i, "image": p, "split": s} for (i, s, p) in items],
                "classification": "C_different_rgb_same_mask",
                "note": "not auto-dropped; possible repeated label geometry",
            })
            for (i, s, p) in items:
                if decisions[i]["decision"] == "KEEP":
                    decisions[i].update({
                        "decision": "REVIEW_MASK_ONLY_DUPLICATE",
                        "issue_type": f"{label}_cross_split_same_geometry",
                        "duplicate_group_id": f"{label}:{digest[:12]}",
                        "conflicting_split": ",".join(sorted(splits)),
                        "reason": "mask-only cross-split collision, different RGB; review (class C)",
                    })

    before_counts = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    cleaned_rows = [r for i, r in enumerate(rows) if i not in drop_set]
    after_counts = {s: sum(1 for r in cleaned_rows if r["split"] == s) for s in SPLITS}

    clean_path = out_dir / "manifest_clean.jsonl"
    clean_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned_rows) + "\n")
    sha = _sha256_file(clean_path)
    (out_dir / "manifest_clean.sha256").write_text(f"{sha}  manifest_clean.jsonl\n")

    g0_errors = audit(str(clean_path), resize_size=resize_size, normal_policy="none")
    # B+ constraint #10 does NOT require raw-mask / decoded-binary-mask reuse to be
    # zero: class-C mask-only collisions are intentionally RETAINED and reported as
    # repeated_label_geometry. Those specific audit findings are tolerated here; every
    # other finding (lineage, decoded-RGB, pair, spatial/alignment, resized-empty) blocks.
    _mask_reuse = ("raw-mask reused across splits", "decoded-binary-mask reused across splits")
    tolerated = [e for e in g0_errors if any(t in e for t in _mask_reuse)]
    blocking_errors = [e for e in g0_errors if e not in tolerated]
    g0_pass = (len(blocking_errors) == 0) and (len(block_groups) == 0)
    status = "PASS" if g0_pass else ("BLOCKED" if block_groups else "FAIL")

    report = {
        "input": str(Path(input_path).resolve()),
        "cleaned_manifest": str(clean_path.resolve()),
        "cleaned_manifest_sha256": sha,
        "status": status, "gate0_pass": g0_pass, "gate0_error_count": len(g0_errors),
        "block_val_test_groups": [{"kind": k, "digest": d, "splits": sp} for (k, d, sp) in block_groups],
        "finding_count": finding_count, "unique_group_count": unique_group_count,
        "unique_affected_row_count": len(unique_affected_rows),
        "decisions_summary": {d: sum(1 for x in decisions if x["decision"] == d) for d in DECISIONS},
        "before_split_counts": before_counts, "after_split_counts": after_counts,
        "rows_removed": {s: before_counts[s] - after_counts[s] for s in SPLITS},
        "repeated_label_geometry_count": len(repeated_geometry),
        "repeated_label_geometry": repeated_geometry,
        "expected_policy": {"test_removed": 0, "val_removed": 0,
                             "note": "nonzero only if a val/test conflict forced STOP"},
    }
    (out_dir / "manifest_clean_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    csv_path = out_dir / "manifest_clean_decisions.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "row_id", "image", "mask", "original_split", "lineage_id_before",
            "lineage_id_after", "issue_type", "duplicate_group_id",
            "conflicting_split", "decision", "reason", "retained_counterpart"])
        w.writeheader()
        for d in decisions:
            w.writerow(d)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resize-size", type=int, default=256)
    a = ap.parse_args()
    report = run(a.input, a.out_dir, a.resize_size)
    print(json.dumps({k: report[k] for k in (
        "status", "gate0_pass", "finding_count", "unique_group_count",
        "unique_affected_row_count", "rows_removed", "decisions_summary",
        "repeated_label_geometry_count")}, indent=2))
    if report["gate0_error_count"]:
        print("G0 residual errors (first 20):")
        # gate0_errors not returned; re-run audit for display
        from oasis_cycle_aosk.audit import audit as _audit
        for e in _audit(report["cleaned_manifest"], resize_size=256, normal_policy="none")[:20]:
            print("  ", e)
    sys.exit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
