#!/usr/bin/env python3
"""Deterministic pre-model leakage/duplicate repair for a crack manifest.

Policy:
- normalize split-qualified lineage IDs;
- preserve the highest-priority evaluation split (test > val > train);
- repair cross-split lineage/RGB/pair/non-empty-mask leakage by dropping lower-priority rows;
- remove exact decoded-RGB and decoded image-mask pair duplicates within a split;
- never mutate raw data;
- native-empty targets are left for build_cleaneval_v1.py to resolve conservatively.
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
                return lineage[len(prefix) :]
    return lineage


def run(input_path, out_dir, resize_size=256):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in Path(input_path).read_text().splitlines()
        if line.strip()
    ]

    decisions = []
    for i, row in enumerate(rows):
        if row.get("split") not in SPLITS:
            raise ValueError(f"unsupported split at row {i}: {row.get('split')!r}")
        before = str(row.get("lineage_id", ""))
        after = strip_lineage(before)
        row["lineage_id"] = after
        decisions.append(
            {
                "row_id": i,
                "image": row.get("image"),
                "mask": row.get("mask"),
                "original_split": row.get("split"),
                "lineage_id_before": before,
                "lineage_id_after": after,
                "decision": "FIX_LINEAGE_FORMAT" if before != after else "KEEP",
                "issue_type": "split_qualified_lineage" if before != after else "",
                "group_id": "",
                "conflicting_splits": "",
                "retained_split": "",
                "reason": "stripped split prefix" if before != after else "",
            }
        )

    meta = []
    for row in rows:
        rgb = _decoded_rgb_sha256(row["image"])
        pair = None
        mask_digest = None
        mask_fg = None
        if row.get("is_normal") is not True and row.get("mask"):
            binary = _decoded_binary_mask(row["mask"])
            mask_fg = int(binary.sum())
            mask_digest = _array_sha256(binary, "binary-mask")
            pair = _pair_sha256(rgb, mask_digest)
        meta.append(
            {"rgb": rgb, "pair": pair, "mask": mask_digest, "mask_fg": mask_fg}
        )

    drop = set()
    findings = []
    seen_groups = set()

    def mark_drop(index, decision, kind, group_key, splits, retained_split, reason):
        drop.add(index)
        d = decisions[index]
        if d["decision"] in ("KEEP", "FIX_LINEAGE_FORMAT"):
            d["decision"] = decision
            d["issue_type"] = kind
            d["group_id"] = f"{kind}:{str(group_key)[:16]}"
            d["conflicting_splits"] = ",".join(sorted(splits))
            d["retained_split"] = retained_split
            d["reason"] = reason

    def apply_group(kind, group_key, indices, dedupe_within_split=False):
        active = [i for i in indices if i not in drop]
        if len(active) <= 1:
            return
        signature = (kind, str(group_key), tuple(sorted(active)), dedupe_within_split)
        if signature in seen_groups:
            return
        seen_groups.add(signature)

        removed = []
        splits = {rows[i]["split"] for i in active}
        retained_split = None
        if len(splits) > 1:
            retained_split = max(splits, key=lambda s: PRIORITY[s])
            for i in active:
                if rows[i]["split"] != retained_split:
                    mark_drop(
                        i,
                        "DROP_LOWER_PRIORITY_SPLIT",
                        kind,
                        group_key,
                        splits,
                        retained_split,
                        f"{kind} spans splits; preserve {retained_split} under test>val>train policy",
                    )
                    removed.append(i)

        if dedupe_within_split:
            by_split = defaultdict(list)
            for i in active:
                if i not in drop:
                    by_split[rows[i]["split"]].append(i)
            for split, members in by_split.items():
                if len(members) <= 1:
                    continue
                keep = min(members)
                for i in sorted(members):
                    if i == keep:
                        continue
                    mark_drop(
                        i,
                        "DROP_EXACT_DUPLICATE_SAME_SPLIT",
                        kind,
                        group_key,
                        {split},
                        split,
                        f"exact {kind} duplicate within {split}; keep earliest row {keep}",
                    )
                    removed.append(i)

        if removed:
            findings.append(
                {
                    "kind": kind,
                    "group": str(group_key),
                    "splits": sorted(splits),
                    "retained_split": retained_split,
                    "rows": list(indices),
                    "removed_rows": sorted(set(removed)),
                }
            )

    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["lineage_id"]].append(i)
    for key, idxs in groups.items():
        apply_group("lineage", key, idxs)

    groups = defaultdict(list)
    for i, item in enumerate(meta):
        groups[item["rgb"]].append(i)
    for key, idxs in groups.items():
        if len({str(Path(rows[i]["image"]).resolve()) for i in idxs}) > 1:
            apply_group("decoded_rgb", key, idxs, dedupe_within_split=True)

    groups = defaultdict(list)
    for i, item in enumerate(meta):
        if item["pair"]:
            groups[item["pair"]].append(i)
    for key, idxs in groups.items():
        apply_group("decoded_pair", key, idxs, dedupe_within_split=True)

    groups = defaultdict(list)
    for i, item in enumerate(meta):
        if item["mask"] and item["mask_fg"] and item["mask_fg"] > 0:
            groups[item["mask"]].append(i)
    for key, idxs in groups.items():
        apply_group("nonempty_binary_mask", key, idxs)

    cleaned = [row for i, row in enumerate(rows) if i not in drop]
    clean_path = out_dir / "manifest_clean.jsonl"
    clean_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in cleaned) + "\n"
    )

    errors = audit(clean_path, resize_size=int(resize_size), normal_policy="none")
    tolerated_tokens = (
        "crack-positive row has native-empty mask",
        "raw-mask reused across splits",
        "decoded-binary-mask reused across splits",
    )
    blocking = [e for e in errors if not any(t in e for t in tolerated_tokens)]

    before = {s: sum(row["split"] == s for row in rows) for s in SPLITS}
    after = {s: sum(row["split"] == s for row in cleaned) for s in SPLITS}
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
        "policy": "test > val > train; exact RGB/pair duplicates deduplicated within split",
        "requires_empty_target_resolution": any(
            item["mask_fg"] == 0 for item in meta if item["mask_fg"] is not None
        ),
    }
    (out_dir / "manifest_clean_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    (out_dir / "manifest_clean.sha256").write_text(
        f"{report['cleaned_manifest_sha256']}  manifest_clean.jsonl\n"
    )
    (out_dir / "leakage_groups.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False)
    )

    fields = [
        "row_id",
        "image",
        "mask",
        "original_split",
        "lineage_id_before",
        "lineage_id_after",
        "decision",
        "issue_type",
        "group_id",
        "conflicting_splits",
        "retained_split",
        "reason",
    ]
    with (out_dir / "manifest_clean_decisions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decisions)

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
