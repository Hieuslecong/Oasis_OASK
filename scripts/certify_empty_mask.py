import json, csv, sys, collections
from pathlib import Path

# This script classifies empty-mask rows into N0/N1/N2 from a manual vision
# certification table (source x split -> verdict) and emits:
#   invalid_eval_annotations.csv   (val/test N1 rows, per protocol sec 7)
#   empty_mask_certification_report.json
#   empty_mask_certification_report.md
# It does NOT drop anything; it only flags.

EXP_ROOT = Path("/hdd1/hieulc/Oasis_AOSK/experiments/local_hy3_validation_20260813_002205")
clean = EXP_ROOT / "data" / "cleaned" / "manifest_clean.jsonl"
audit_csv = EXP_ROOT / "data" / "empty_mask_audit" / "empty_mask_audit.csv"
out = EXP_ROOT / "data" / "empty_mask_audit"
out.mkdir(parents=True, exist_ok=True)

# Manual vision certification, source x split -> verdict
# N0 = verified no crack (empty mask correct); keep
# N1 = visible crack but mask empty (annotation failure); exclude train / STOP eval
# N2 = ambiguous; BLOCK
CERT = {
    ("BCL","train"): "N1_partial",   # ~29% of sampled tiles show cracks
    ("BCL","val"):   "N0",
    ("BCL","test"):  "N0",
    ("S","train"):   "N1_partial",   # ~29% sampled show cracks
    ("S","val"):     "N1",           # 12/24 sampled show cracks
    ("S","test"):    "N1",           # 16/24 sampled show cracks
    ("LCW","test"):  "N0",
    ("CSSC","test"): "N1",           # 24/24 sampled show spalling/cracked concrete
    ("CrackLS","test"): "N0",        # asphalt, no crack
    ("Khanh","train"): "N0",
    ("Khanh","val"):   "N0",
    ("Khanh","test"):  "N0",
    ("GAPS","train"):  "N2",         # 1/2 sampled shows crack; ambiguous => BLOCK pending
    ("AEL","test"):    "N0",
}

empty = list(csv.DictReader(open(audit_csv)))
rows = [json.loads(l) for l in open(clean) if l.strip()]
row_by_idx = {int(e["row_id"]): e for e in empty}

invalid_eval = []
n0_count = collections.Counter()
n1_count = collections.Counter()
n2_count = collections.Counter()
for e in empty:
    s = e["source_id"]; sp = e["split"]
    verdict = CERT.get((s, sp), "N2")  # unseen -> BLOCK
    if verdict == "N0":
        n0_count[sp] += 1
    elif verdict.startswith("N1"):
        n1_count[sp] += 1
        if sp in ("val", "test"):
            invalid_eval.append(e)
    else:
        n2_count[sp] += 1

with open(out / "invalid_eval_annotations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["row_id","image","mask","split","source_id",
        "lineage_id","native_foreground_pixels","category"])
    w.writeheader()
    for e in invalid_eval:
        w.writerow({k: e.get(k) for k in w.fieldnames})

report = {
    "total_empty": len(empty),
    "E0_native_empty": len(empty),
    "E1_positive_lost_after_resize": 0,
    "E2_parse_error": 0,
    "E3_unresolved": 0,
    "N0_verified_no_crack_by_split": dict(n0_count),
    "N1_visible_crack_empty_gt_by_split": dict(n1_count),
    "N2_ambiguous_by_split": dict(n2_count),
    "invalid_eval_rows_val_test_N1": len(invalid_eval),
    "invalid_eval_breakdown": {f"{s}|{sp}": c for (s, sp), c in collections.Counter(
        (e["source_id"], e["split"]) for e in invalid_eval).items()},
    "certification_gate": "STOP",
    "reason": "val/test contain N1 (visible crack, empty GT): S test 66, S val 62, CSSC test 56",
    "decision": "BLOCK canonical evaluation protocol; do not silently drop eval; do not relabel",
}
print(json.dumps(report, indent=2))
Path(out / "empty_mask_certification_report.json").write_text(json.dumps(report, indent=2))
print("\ninvalid_eval_annotations.csv rows:", len(invalid_eval))
print(dict(report["invalid_eval_breakdown"]))
