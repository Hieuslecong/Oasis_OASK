#!/usr/bin/env python3
"""Validate row-level empty-GT review decisions.

This tool deliberately does not infer a verdict from source, split, filename
family, or a sampled subset. Only explicit row-level decisions are accepted.
Unreviewed rows remain unreviewed; the CleanEval builder excludes them
conservatively rather than silently treating them as N0 or N1.
"""
import argparse
import csv
import json
from pathlib import Path

ALLOWED = {"N0", "N1", "N2", "N3"}


def norm_verdict(value):
    value = str(value or "").strip().upper()
    for prefix in ALLOWED:
        if value == prefix or value.startswith(prefix + "_"):
            return prefix
    raise ValueError(f"invalid verdict {value!r}; expected N0/N1/N2/N3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", required=True)
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    audit = list(csv.DictReader(open(args.audit_csv, newline="")))
    review = list(csv.DictReader(open(args.review_csv, newline="")))
    audit_images = {str(Path(r["image"]).resolve()): r for r in audit}

    normalized = []
    seen = set()
    for r in review:
        if not r.get("image"):
            raise ValueError("every review row must contain image")
        key = str(Path(r["image"]).resolve())
        if key in seen:
            raise ValueError(f"duplicate review decision for {key}")
        seen.add(key)
        if key not in audit_images:
            raise ValueError(f"review row not present in empty-mask audit: {key}")
        verdict = norm_verdict(r.get("verdict"))
        reviewer = str(r.get("reviewer") or "").strip()
        reason = str(r.get("reason") or "").strip()
        if not reviewer or not reason:
            raise ValueError(f"row-level certification requires reviewer and reason: {key}")
        normalized.append({
            "image": key,
            "mask": audit_images[key].get("mask"),
            "split": audit_images[key].get("split"),
            "source_id": audit_images[key].get("source_id"),
            "lineage_id": audit_images[key].get("lineage_id"),
            "verdict": verdict,
            "reviewer": reviewer,
            "reason": reason,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "mask", "split", "source_id", "lineage_id", "verdict", "reviewer", "reason"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(normalized)

    report = {
        "audit_empty_rows": len(audit),
        "row_level_reviewed": len(normalized),
        "unreviewed": len(audit) - len(normalized),
        "counts": {v: sum(r["verdict"] == v for r in normalized) for v in sorted(ALLOWED)},
        "policy": "row-level only; no source×split inference",
    }
    out.with_suffix(out.suffix + ".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
