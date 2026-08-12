import argparse, json
from pathlib import Path
from PIL import Image

REQUIRED = {"image", "split", "source_id", "lineage_id", "is_normal"}

def audit(path, allow_debug_no_test_normals=False, test_split="test", require_source_disjoint=False, require_normal=True):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    errors, lineage_splits, source_splits, image_seen = [], {}, {}, set()
    for i, r in enumerate(rows):
        missing = REQUIRED - r.keys()
        if missing: errors.append(f"row {i}: missing {sorted(missing)}"); continue
        lineage_splits.setdefault(r["lineage_id"], set()).add(r["split"])
        source_splits.setdefault(r["source_id"], set()).add(r["split"])
        image_key = str(Path(r["image"]).resolve())
        if image_key in image_seen: errors.append(f"duplicate image path: {image_key}")
        image_seen.add(image_key)
        if not Path(r["image"]).exists(): errors.append(f"row {i}: missing image file"); continue
        if not r["is_normal"]:
            if not r.get("mask") or not Path(r["mask"]).exists(): errors.append(f"row {i}: missing cracked mask"); continue
            if Image.open(r["image"]).size != Image.open(r["mask"]).size: errors.append(f"row {i}: native-resolution mismatch")
    for lineage, splits in lineage_splits.items():
        if len(splits) > 1: errors.append(f"lineage leakage: {lineage} in {sorted(splits)}")
    if require_source_disjoint:
        for source, splits in source_splits.items():
            if len(splits) > 1: errors.append(f"source leakage: {source} in {sorted(splits)}")
    required_splits = ("train", "val", test_split)
    for split in required_splits:
        is_test = split == test_split
        if not any(r.get("split") == split and bool(r.get("is_normal")) for r in rows) and not (is_test and allow_debug_no_test_normals) and require_normal: errors.append(f"{split}: no normal sample")
    return errors

def main():
    p = argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--allow-debug-no-test-normals", action="store_true"); p.add_argument("--test-split", default="test"); p.add_argument("--require-source-disjoint", action="store_true"); a = p.parse_args()
    errors = audit(a.manifest, a.allow_debug_no_test_normals, a.test_split, a.require_source_disjoint)
    if errors:
        print("G0 FAIL"); print("\n".join(errors)); raise SystemExit(2)
    label = "G0 DEBUG-ONLY PASS" if a.test_split != "test" or a.allow_debug_no_test_normals else "G0 PASS"
    print(label)

if __name__ == "__main__": main()
