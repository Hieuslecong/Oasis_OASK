# OmniCrack30k-CleanEval-v1 — Work Report (Gate 0 certified)

**Date:** 2026-08-13
**Repo:** Oasis_OASK · **Branch:** `agent/normal-rgb-oasis-rc-v2` · **Commit:** `a8bc85b`
**Artifacts root:** `experiments/local_hy3_validation_20260813_002205/data/cleaneval_v1/`

## What was done

Built **OmniCrack30k-CleanEval-v1**, a pre-model data-integrity-certified
derivative of the OmniCrack30k canonical manifest. The canonical manifest is
left **immutable**; this is a derived benchmark.

Two policies were applied (mixed certified-repair):

1. **TRAIN** — Quarantine uncertified empty-GT rows from BCL/S/S2DS/GAPS
   train splits (`QUARANTINE_UNCERTIFIED_EMPTY_GT`). These rows are NOT
   claimed all-N1; they are merely untrustworthy for official training.
   Raw data was not modified or deleted — quarantined rows are kept in a CSV.
2. **EVAL** — Exclude only the 184 individually row-level-certified
   `N1_VISIBLE_TARGET_DEFECT_EMPTY_GT` rows (RGB+GT audit only, no model
   prediction). No whole source (S/S2DS/CSSC) was dropped.

Additionally enforced:
- **Non-empty mask reuse exclusion** (7 rows): a real crack mask reused across
  splits is a leakage defect.
- **Lineage leakage exclusion** (133 eval rows = 119 train↔test + 14 val↔test):
  same `lineage_id`/`image_basename` appearing in multiple splits.

## Authoritative Gate 0 result

Audit run: `python -m oasis_cycle_aosk.audit --manifest <full derived>
--resize-size 256 --normal-policy none`

```
G0 PASS
full manifest SHA: 6b960074f32834ea194f137eec8c40b57afee382719d61e8c9f9698cb5ea672f
  native-empty:          0
  reused across splits:  0
  lineage leakage:       0
```

Re-verified independently (second run, EXIT=0) — live full manifest SHA
matches the frozen `g0_full_manifest_sha` byte-for-byte.

## Derived benchmark counts

| Split | Rows |
|-------|------|
| train | 19,187 |
| val   | 3,199  |
| test  | 4,004  |
| **total** | **26,390** |
| certified true-negative (N0) empties kept | 905 |

## Exclusion breakdown

| Category | Count | File |
|----------|-------|------|
| train quarantined (contam/empty) | 2,958 | `cleaneval_v1_quarantine.csv` |
| eval N1 certified excluded | 184 | `cleaneval_v1_exclusions.csv` |
| eval uncert (val / test) | 62 / 122 | same |
| non-empty mask reuse excluded | 7 | `cleaneval_v1_exclusions.csv` |
| lineage leakage eval excluded | 133 (119 train↔test, 14 val↔test) | `lineage_leakage_exclusions.csv` |
| reannotation queue (v2 candidate) | 184 | `reannotation_queue_184.csv` |

Eval exclusion by source: `S|val` 62, `S|test` 66, `CSSC|test` 56 (total 184).
Original eval: val 3277 → 3215, test 4251 → 4129.

## Code changes (commit a8bc85b, 12 files, +1078/-5)

Core fix:
- `src/oasis_cycle_aosk/audit.py` — certified-empty masks (`verified_no_crack`)
  no longer flagged as native-empty errors; mask-reuse gate fires only on
  **non-empty** crack masks reused across splits.
- `tests/test_audit.py` — added cross-split certified-empty regression test;
  **9/9 pass**.
- `tests/test_normal_rgb.py` — normal-RGB source coverage extended.

Pipeline (new scripts + tests):
`build_cleaneval_v1.py`, `certify_empty_mask.py`, `clean_manifest.py`,
`audit_empty_mask.py`, `make_empty_contact_sheets.py`,
`run_validation_arms.sh`, `run_validation_eval.sh`,
`add_normal_rgb_to_manifest.py`, `tests/test_clean_manifest.py`.

## Artifacts

Manifest: `manifest_clean_train.jsonl`, `manifest_cleaneval_v1.jsonl`,
`manifest_cleaneval_v1_full.jsonl` (concatenated, audited).
Provenance: `benchmark_freeze.json`, `build_provenance.json`,
`lineage_exclusion_provenance.json`, `cleaneval_v1_report.json`,
`cleaneval_v1.sha256`.
Copies of the key provenance files are also under `reports/cleaneval_v1/`.

## Status

Gate 0 **PASS** — ready for 4-arm smoke / validation arms on the cleaned
benchmark. Test-metric firewall: CLOSED (no N2/N3 remain in eval).
