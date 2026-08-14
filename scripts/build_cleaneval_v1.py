#!/usr/bin/env python3
"""Build a fail-closed OmniCrack30k CleanEval derivative.

Scientific policy:
- raw/canonical files are never modified;
- lineage/RGB/mask leakage must already be repaired by clean_manifest.py;
- native-empty crack-source targets are kept only with an explicit row-level N0
  certification; N1/N2/N3/unreviewed rows are excluded conservatively;
- no model prediction or test metric is used;
- hashes are generated from the exact final files in one run.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from oasis_cycle_aosk.audit import audit  # noqa: E402

SPLITS = ("train", "val", "test")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_foreground(path):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return int((arr > 127).sum())


def verdict_code(value):
    value = str(value or "").strip().upper()
    for code in ("N0", "N1", "N2", "N3"):
        if value == code or value.startswith(code + "_"):
            return code
    return None


def load_certifications(path):
    if not path:
        return {}
    rows = list(csv.DictReader(open(path, newline="")))
    result = {}
    for r in rows:
        key = str(Path(r["image"]).resolve())
        code = verdict_code(r.get("verdict"))
        if code is None:
            raise ValueError(f"invalid certification verdict for {key}")
        if key in result:
            raise ValueError(f"duplicate certification for {key}")
        result[key] = {**r, "verdict": code}
    return result


def build(input_path, out_dir, certification_csv=None, resize_size=256):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cert = load_certifications(certification_csv)
    rows = [json.loads(l) for l in Path(input_path).read_text().splitlines() if l.strip()]

    kept = []
    exclusions = []
    used_cert = set()

    for r in rows:
        sp = r.get("split")
        if sp not in SPLITS:
            continue
        if r.get("is_normal") is True:
            kept.append(dict(r))
            continue
        mp = r.get("mask")
        if not mp or not Path(mp).exists():
            raise RuntimeError(f"missing mask for crack-source row: {r.get('image')}")
        fg = mask_foreground(mp)
        if fg > 0:
            kept.append(dict(r))
            continue

        key = str(Path(r["image"]).resolve())
        decision = cert.get(key)
        if decision and decision["verdict"] == "N0":
            used_cert.add(key)
            r2 = dict(r)
            r2["empty_target_status"] = "verified_no_crack"
            r2["empty_target_certification"] = {
                "verdict": "N0",
                "reviewer": decision.get("reviewer"),
                "reason": decision.get("reason"),
            }
            kept.append(r2)
        else:
            verdict = decision["verdict"] if decision else "UNREVIEWED"
            if decision:
                used_cert.add(key)
            exclusions.append({
                "image": r.get("image"),
                "mask": mp,
                "split": sp,
                "source_id": r.get("source_id"),
                "lineage_id": r.get("lineage_id"),
                "verdict": verdict,
                "reason": "native-empty GT not explicitly row-certified N0",
            })

    full = out / "manifest_cleaneval_v1_full.jsonl"
    train = out / "manifest_clean_train.jsonl"
    evalp = out / "manifest_cleaneval_v1.jsonl"

    def write_jsonl(path, selected):
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + ("\n" if selected else ""))

    write_jsonl(full, kept)
    write_jsonl(train, [r for r in kept if r.get("split") == "train"])
    write_jsonl(evalp, [r for r in kept if r.get("split") in ("val", "test")])

    ex_path = out / "cleaneval_v1_exclusions.csv"
    fields = ["image", "mask", "split", "source_id", "lineage_id", "verdict", "reason"]
    with ex_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(exclusions)

    errors = audit(full, resize_size=int(resize_size), normal_policy="none")
    if errors:
        (out / "gate0_errors.json").write_text(json.dumps(errors, indent=2))
        raise RuntimeError("G0 FAIL:\n" + "\n".join(errors[:50]))

    hashes = {
        "manifest_clean_train.jsonl": sha256_file(train),
        "manifest_cleaneval_v1.jsonl": sha256_file(evalp),
        "manifest_cleaneval_v1_full.jsonl": sha256_file(full),
        "cleaneval_v1_exclusions.csv": sha256_file(ex_path),
    }
    (out / "cleaneval_v1.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))

    split_counts = Counter(r.get("split") for r in kept)
    report = {
        "benchmark_name": "OmniCrack30k-CleanEval-v1",
        "status": "PASS",
        "source_manifest": str(Path(input_path).resolve()),
        "source_rows": len(rows),
        "kept_rows": len(kept),
        "split_counts": {s: int(split_counts.get(s, 0)) for s in SPLITS},
        "empty_target_excluded": len(exclusions),
        "certified_n0_kept": sum(r.get("empty_target_status") == "verified_no_crack" for r in kept),
        "certification_csv": str(Path(certification_csv).resolve()) if certification_csv else None,
        "unused_certification_rows": sorted(set(cert) - used_cert),
        "hashes": hashes,
        "policy": "fail-closed; explicit row-level N0 only; N1/N2/N3/unreviewed native-empty rows excluded",
        "test_metrics_opened": False,
    }
    (out / "benchmark_freeze.json").write_text(json.dumps(report, indent=2))
    (out / "build_provenance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--certification-csv", default=None)
    ap.add_argument("--resize-size", type=int, default=256)
    args = ap.parse_args()
    build(args.input, args.out_dir, args.certification_csv, args.resize_size)


if __name__ == "__main__":
    main()
