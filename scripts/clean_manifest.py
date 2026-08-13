#!/usr/bin/env python3
"""Deterministic pre-model leakage repair for a crack manifest.

Policy:
- normalize split-qualified lineage IDs;
- preserve the highest-priority evaluation split (test > val > train);
- if a lineage, decoded RGB, decoded RGB+mask pair, or non-empty binary mask
  appears across splits, remove lower-priority split rows from the derived copy;
- never mutate raw data;
- native-empty targets are not certified here and are left for
  build_cleaneval_v1.py to resolve conservatively.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oasis_cycle_aosk.audit import (  # noqa: E402
    _array_sha256,
    _decoded_binary_mask,
    _decoded_rgb_sha256,
    _pair_sha256,
    _sha256_file,
    audit,
)

PRIORITY = {"test": 3, "val": 2, "train": 1}
SPLITS = ("train", "val", "test")


def strip_lineage(lineage: str) -> str:
    lineage = str(lineage)
    for split in SPLITS:
        for sep in ("::", ":"):
            prefix = split + sep
            if lineage.startswith(prefix):
                return lineage[len(prefix):]
    return lineage


def run(input_path, out_dir, resize_size=256):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(input_path).read_text().splitlines() if l.strip()]

    decisions = []
    for i, r in enumerate(rows):
        if r.get("split") not in SPLITS:
            raise ValueError(f"unsupported split at row {i}: {r.get('split')!r}")
        before = str(r.get("lineage_id", ""))
        after = strip_lineage(before)
        r["lineage_id"] = after
        decisions.append({
            "row_id": i,
            "image": r.get("image"),
            "mask": r.get("mask"),
            "original_split": r.get("split"),
            "lineage_id_before": before,
            "lineage_id_after": after,
            "decision": "FIX_LINEAGE_FORMAT" if before != after else "KEEP",
            "issue_type": "split_qualified_lineage" if before != after else "",
            "group_id": "",
            "conflicting_splits": "",
            "retained_split": "",
            "reason": "stripped split prefix" if before != after else "",
        })

    meta = []
    for r in rows:
        rgb = _decoded_rgb_sha256(r["image"])
        pair = None
        mask_digest = None
        mask_fg = None
        if r.get("is_normal") is not True and r.get("mask"):
            bm = _decoded_binary_mask(r["mask"])
            mask_fg = int(bm.sum())
            mask_digest = _array_sha256(bm, "binary-mask")
            pair = _pair_sha256(rgb, mask_digest)
        meta.append({"rgb": rgb, "pair": pair, "mask": mask_digest, "mask_fg": mask_fg})

    drop = set()
    findings = []
    seen_groups = set()

    def apply_group(kind, group_key, indices):
        splits = {rows[i]["split"] for i in indices}
        if len(splits) <= 1:
            return
        signature = (kind, str(group_key), tuple(sorted(indices)))
        if signature in seen_groups:
            return
        seen_groups.add(signature)
        keep_split = max(splits, key=lambda s: PRIORITY[s])
        removed = []
        for i in indices:
            if rows[i]["split"] != keep_split:
                drop.add(i)
                removed.append(i)
                d = decisions[i]
                if d["decision"] in ("KEEP", "FIX_LINEAGE_FORMAT"):
                    d["decision"] = "DROP_LOWER_PRIORITY_SPLIT"
                    d["issue_type"] = kind
                    d["group_id"] = f"{kind}:{str(group_key)[:16]}"
                    d["conflicting_splits"] = ",".join(sorted(splits))
                    d["retained_split"] = keep_split
                    d["reason"] = f"{kind} spans splits; preserve {keep_split} under test>val>train policy"
        findings.append({
            "kind": kind,
            "group": str(group_key),
            "splits": sorted(splits),
            "retained_split": keep_split,
            "rows": list(indices),
            "removed_rows": removed,
        })

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["lineage_id"]].append(i)
    for key, idxs in groups.items():
        apply_group("lineage", key, idxs)

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["rgb"]].append(i)
    for key, idxs in groups.items():
        if len({str(Path(rows[i]["image"]).resolve()) for i in idxs}) > 1:
            apply_group("decoded_rgb", key, idxs)

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        if m["pair"]:
            groups[m["pair"]].append(i)
    for key, idxs in groups.items():
        apply_group("decoded_pair", key, idxs)

    groups = defaultdict(list)
    for i, m in enumerate(meta):
        if m["mask"] and m["mask_fg"] and m["mask_fg"] > 0:
            groups[m["mask"]].append(i)
    for key, idxs in groups.items():
        apply_group("nonempty_binary_mask", key, idxs)

    cleaned = [r for i, r in enumerate(rows) if i not in drop]
    clean_path = out_dir / "manifest_clean.jsonl"
    clean_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned) + "\n")

    errors = audit(clean_path, resize_size=int(resize_size), normal_policy="none")
    tolerated_tokens = (
        "crack-positive row has native-empty mask",
        "raw-mask reused across splits",
        "decoded-binary-mask reused across splits",
    )
    blocking = [e for e in errors if not any(t in e for t in tolerated_tokens)]

    before = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    after = {s: sum(r["split"] == s for r in cleaned) for s in SPLITS}
    report = {
        "input": str(Path(input_path).resolve()),
        "cleaned_manifest": str(clean_path.resolve()),
        "cleaned_manifest_sha256": _sha256_file(clean_path),
        "status": "PASS" if not blocking else "FAIL",
        "preclean_blocking_errors": blocking,
        "tolerated_empty_target_errors": len(errors) - len(blocking),
        "finding_count": len(findings),
        "unique_removed_rows": len(drop),
        "before_split_counts": before,
        "after_split_counts": after,
        "rows_removed": {s: before[s] - after[s] for s in SPLITS},
        "policy": "test > val > train; preserve evaluation, drop lower-priority leakage",
        "requires_empty_target_resolution": any(m["mask_fg"] == 0 for m in meta if m["mask_fg"] is not None),
    }
    (out_dir / "manifest_clean_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "manifest_clean.sha256").write_text(f"{report['cleaned_manifest_sha256']}  manifest_clean.jsonl\n")
    (out_dir / "leakage_groups.json").write_text(json.dumps(findings, indent=2, ensure_ascii=False))

    fields = [
        "row_id", "image", "mask", "original_split", "lineage_id_before",
        "lineage_id_after", "decision", "issue_type", "group_id",
        "conflicting_splits", "retained_split", "reason",
    ]
    with (out_dir / "manifest_clean_decisions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(decisions)

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resize-size", type=int, default=256)
    args = ap.parse_args()
    report = run(args.input, args.out_dir, args.resize_size)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
