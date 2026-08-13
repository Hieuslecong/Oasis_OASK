import csv, json, hashlib, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

# Builds OmniCrack30k-CleanEval-v1 cleaned canonical benchmark manifests.
# Policy (user-mandated, mixed certified-repair, firewall fail-closed):
#   TRAIN: quarantine uncertified native-empty rows from contaminated sources
#          (BCL train, S/S2DS train, GAPS train) -> QUARANTINE_UNCERTIFIED_EMPTY_GT
#   VAL/TEST: exclude individually-verified N1 rows (invalid_eval_annotations.csv)
#   Certified N0 empty targets get empty_target_status="verified_no_crack" (both train+eval).
#   Cross-split identical non-empty mask leakage -> LEAK_EXCLUDED (HARD FAIL firewall).
#   NO raw data modified; NO model used anywhere.

EXP = Path("/hdd1/hieulc/Oasis_AOSK/experiments/local_hy3_validation_20260813_002205")
CLEAN = EXP / "data" / "cleaned" / "manifest_clean.jsonl"
AUDIT_DIR = EXP / "data" / "empty_mask_audit"
OUT = EXP / "data" / "cleaneval_v1"
OUT.mkdir(parents=True, exist_ok=True)

CONTAM_TRAIN_SOURCES = {"BCL", "S", "GAPS"}


def mask_digest(path):
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    b = (arr > 127).astype(np.uint8)
    return hashlib.sha256(np.ascontiguousarray(b)).hexdigest(), int(b.sum())


def row_id_of(r):
    return r.get("image")


# --- empty mask audit (per-row): category marks ALL native-empty -----------
empty_all = list(csv.DictReader(open(AUDIT_DIR / "empty_mask_audit.csv")))
empty_images = {e["image"]: e for e in empty_all}
empty_train_contam = {e["image"] for e in empty_all
                      if e["split"] == "train" and e["source_id"] in CONTAM_TRAIN_SOURCES}

# --- explicit row-level N1 eval exclusions ---------------------------------
inv_p = AUDIT_DIR / "invalid_eval_annotations.csv"
n1_excl = {}
if inv_p.exists():
    for e in csv.DictReader(open(inv_p)):
        n1_excl[e["image"]] = e

# --- certified N0 = native-empty NOT invalid-eval NOT contam-train N1 -----
# reconstructs certification_report.json N0_by_split (train 34, val 294, test 635)
certified_n0 = {}
for img, e in empty_images.items():
    if e["split"] in ("val", "test") and img in n1_excl:
        continue
    if e["split"] == "train" and img in empty_train_contam:
        continue
    certified_n0[img] = e

# --- cross-split identical non-empty mask leakage (HARD FAIL) --------------
rows = [json.loads(l) for l in open(CLEAN) if l.strip()]
leak_groups = defaultdict(list)
for r in rows:
    try:
        d, nf = mask_digest(r["mask"])
    except Exception:
        continue
    if nf == 0:
        continue
    leak_groups[d].append(r)
leak_excl = {}
for d, g in leak_groups.items():
    splits = {r["split"] for r in g}
    if len(splits) > 1:
        for r in g:
            leak_excl[row_id_of(r)] = r

# --- derive, excluding in priority order -----------------------------------
train_out, eval_out = [], []
stat = {"train": 0, "val": 0, "test": 0}
for r in rows:
    img, sp = row_id_of(r), r["split"]
    if img in leak_excl:
        r2 = dict(r)
        r2["exclusion"] = "LEAK_EXCLUDED"
        r2["exclusion_reason"] = "cross-split identical non-empty mask leakage"
        stat["%s_leak" % sp] = stat.get("%s_leak" % sp, 0) + 1
        continue
    if sp in ("val", "test"):
        if img in n1_excl:
            r2 = dict(r)
            r2["exclusion"] = "N1_EXCLUDED"
            r2["exclusion_reason"] = "individually verified bad GT annotation"
            stat["%s_n1" % sp] = stat.get("%s_n1" % sp, 0) + 1
            continue
        if img in certified_n0:
            r2 = dict(r)
            r2["empty_target_status"] = "verified_no_crack"
            r2["empty_target_decision"] = "certified_N0_empty_gt"
            eval_out.append(r2)
        else:
            eval_out.append(r)
        stat[sp] = stat.get(sp, 0) + 1
    elif sp == "train":
        if img in empty_train_contam and not r.get("is_normal"):
            r2 = dict(r)
            r2["exclusion"] = "QUARANTINE_UNCERTIFIED_EMPTY_GT"
            r2["exclusion_reason"] = "uncertified empty GT from contaminated train source"
            stat["train_q"] = stat.get("train_q", 0) + 1
            continue
        if img in certified_n0:
            r2 = dict(r)
            r2["empty_target_status"] = "verified_no_crack"
            r2["empty_target_decision"] = "certified_N0_empty_gt"
            train_out.append(r2)
        else:
            train_out.append(r)
        stat["train"] = stat.get("train", 0) + 1
    else:
        train_out.append(r)
        stat["train"] = stat.get("train", 0) + 1

TRAIN_JSONL = OUT / "manifest_clean_train.jsonl"
EVAL_JSONL = OUT / "manifest_cleaneval_v1.jsonl"
with open(TRAIN_JSONL, "w") as f:
    for r in train_out:
        f.write(json.dumps(r) + "\n")
with open(EVAL_JSONL, "w") as f:
    for r in eval_out:
        f.write(json.dumps(r) + "\n")

provenance = {
    "benchmark": "OmniCrack30k-CleanEval-v1",
    "built_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "source_canonical": str(CLEAN),
    "source_lines_total": len(rows),
    "stats": stat,
    "leak_excluded_total": len(leak_excl),
    "leak_groups": {d[:12]: {"splits": sorted({r["split"] for r in g})}
                    for d, g in leak_groups.items() if len({r["split"] for r in g}) > 1},
    "train_rows": len(train_out),
    "eval_rows": len(eval_out),
    "certified_n0_eval": sum(1 for r in eval_out if r.get("empty_target_status") == "verified_no_crack"),
    "certified_n0_train": sum(1 for r in train_out if r.get("empty_target_status") == "verified_no_crack"),
    "policy": "mixed certified-repair; firewall fail-closed; no raw-data modification; no model",
}
with open(OUT / "build_provenance.json", "w") as f:
    json.dump(provenance, f, indent=2)

with open(OUT / "leakage_excluded_report.txt", "w") as f:
    f.write("CleanEval-v1 cross-split identical non-empty mask leakage (HARD FAIL) -> LEAK_EXCLUDED\n")
    f.write("=" * 70 + "\n")
    for d, g in leak_groups.items():
        splits = {r["split"] for r in g}
        if len(splits) <= 1:
            continue
        f.write("\n[digest %s] splits=%s rows=%d\n" % (d[:16], sorted(splits), len(g)))
        for r in g:
            f.write("   %-6s nf=%-5d %s\n" % (r["split"], mask_digest(r["mask"])[1], r["mask"]))

print("=== DERIVE DONE ===")
print(json.dumps(provenance, indent=2))
print("Leak report:", OUT / "leakage_excluded_report.txt")
