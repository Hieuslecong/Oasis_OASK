#!/usr/bin/env python3
"""Fail-closed manifest leakage audit. No experiment should proceed on FAIL."""
import argparse, hashlib, json
from pathlib import Path

from PIL import Image


def digest(path):
    h = hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def audit(manifest):
    rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    by_name, by_hash, by_source = {}, {}, {}
    issues = []
    for row in rows:
        split = row.get("split"); sid = str(row.get("sample_id", row.get("image")))
        image = Path(row["image"])
        if not image.exists():
            issues.append({"type":"missing_image", "sample_id":sid, "path":str(image)}); continue
        name = image.name; by_name.setdefault(name, []).append((split, sid))
        ih = digest(image); by_hash.setdefault(ih, []).append((split, sid, str(image)))
        source = str(row.get("source_id", sid)); by_source.setdefault(source, []).append((split, sid))
        if row.get("is_normal") is not True and row.get("mask"):
            mask = Path(row["mask"])
            if not mask.exists(): issues.append({"type":"missing_mask", "sample_id":sid, "path":str(mask)})
    for kind, values in (("duplicate_filename", by_name), ("exact_duplicate_image", by_hash), ("source_overlap", by_source)):
        for key, entries in values.items():
            if len({x[0] for x in entries}) > 1:
                issues.append({"type":kind, "key":key, "entries":entries})
    return {"status":"FAIL" if issues else "PASS", "manifest":str(manifest), "rows":len(rows), "issues":issues,
            "limitations":["Perceptual/crop-descendant checks require a configured image index and are not silently inferred from filenames."]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--json", default="reports/data_leakage_report.json"); ap.add_argument("--md", default="reports/data_leakage_report.md"); a=ap.parse_args()
    report=audit(a.manifest); Path(a.json).parent.mkdir(parents=True, exist_ok=True); Path(a.json).write_text(json.dumps(report, indent=2)+"\n")
    lines=[f"# Data Leakage Report\n\nStatus: **{report['status']}**\n", f"Rows: {report['rows']}\n", "\n## Issues\n"]
    lines += [f"- `{item['type']}`: `{item.get('sample_id', item.get('key', ''))}`" for item in report["issues"]] or ["- None detected by implemented checks."]
    lines += ["\n## Limitations\n"] + [f"- {x}" for x in report["limitations"]]
    Path(a.md).parent.mkdir(parents=True, exist_ok=True); Path(a.md).write_text("\n".join(lines)+"\n")
    print(json.dumps(report, indent=2)); raise SystemExit(1 if report["status"] == "FAIL" else 0)

if __name__ == "__main__": main()
