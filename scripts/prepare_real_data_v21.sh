#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT}"
LINEAGE_REGEX="${LINEAGE_REGEX:?set LINEAGE_REGEX to capture independent normal session/parent lineage}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-1337}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CLEAN_DIR="$DATA_ROOT/cleaned_v21"
BENCH_DIR="$DATA_ROOT/cleaneval_v21"
NORMAL_AUDIT="$DATA_ROOT/normal_audit_v21"
NORMAL_EXCLUSIONS="$NORMAL_AUDIT/normal_exclusions.json"
FULL_MANIFEST="$DATA_ROOT/manifest_full_v21.jsonl"
TRAIN_N25="$DATA_ROOT/manifest_trainval_n25_v21.jsonl"
TRAIN_N0="$DATA_ROOT/manifest_trainval_n0_v21.jsonl"
FULL_CERT="$DATA_ROOT/gate0_full_v21.json"
CERT_N25="$DATA_ROOT/gate0_training_n25_v21.json"
CERT_N0="$DATA_ROOT/gate0_training_n0_v21.json"
mkdir -p "$CLEAN_DIR" "$BENCH_DIR" "$NORMAL_AUDIT"

printf '== v2.1 DATA 1/9 clean/leakage repair ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/clean_manifest.py" \
  --input "$CANONICAL_MANIFEST" --out-dir "$CLEAN_DIR" --resize-size 256

printf '== v2.1 DATA 2/9 CleanEval fail-closed ==\n'
ARGS=(--input "$CLEAN_DIR/manifest_clean.jsonl" --out-dir "$BENCH_DIR" --resize-size 256)
if [ -n "${EMPTY_CERTIFICATION_CSV:-}" ]; then ARGS+=(--certification-csv "$EMPTY_CERTIFICATION_CSV"); fi
"$PYTHON" "$PACKAGE_ROOT/scripts/build_cleaneval_v1.py" "${ARGS[@]}"

printf '== v2.1 DATA 3/9 audit external true normals ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/audit_normal_rgb_source.py" \
  --normal-root "$NORMAL_ROOT" \
  --cracked-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl" \
  --repair-exclusions-out "$NORMAL_EXCLUSIONS" \
  --out-dir "$NORMAL_AUDIT" --seed "$SEED"

printf '== v2.1 DATA 4/9 lineage-safe normal train/val/test ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/add_normal_rgb_v21.py" \
  --canonical-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl" \
  --normal-root "$NORMAL_ROOT" \
  --lineage-regex "$LINEAGE_REGEX" \
  --audit-summary "$NORMAL_AUDIT/summary.json" \
  --exclude-file "$NORMAL_EXCLUSIONS" \
  --seed "$SEED" \
  --out "$FULL_MANIFEST"

"$PYTHON" - "$FULL_MANIFEST" <<'PY'
import json,sys
rows=[json.loads(x) for x in open(sys.argv[1]) if x.strip()]
splits={r.get('split') for r in rows}
required={'train','val','test','normal_train','normal_val','normal_test'}
missing=required-splits
if missing: raise SystemExit('v2.1 full manifest missing splits: '+','.join(sorted(missing)))
for split in ('normal_train','normal_val','normal_test'):
    lineages={r.get('lineage_id') for r in rows if r.get('split')==split}
    if not lineages or None in lineages: raise SystemExit(f'{split} missing lineage')
print('NORMAL_SPLITS_OK')
PY

printf '== v2.1 DATA 5/9 full benchmark Gate0 ==\n'
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$FULL_MANIFEST" --resize-size 256 \
  --normal-policy train_and_aux_val --required-splits train val test \
  --certificate-out "$FULL_CERT" --certificate-scope full_benchmark

printf '== v2.1 DATA 6/9 N25 training view ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/build_training_view.py" --input "$FULL_MANIFEST" --out "$TRAIN_N25"

printf '== v2.1 DATA 7/9 N25 Gate0 ==\n'
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$TRAIN_N25" --resize-size 256 \
  --normal-policy train_and_aux_val --required-splits train val \
  --certificate-out "$CERT_N25" --certificate-scope training_view \
  --parent-full-certificate "$FULL_CERT"

printf '== v2.1 DATA 8/9 N0 training view ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/build_training_view.py" --input "$FULL_MANIFEST" --out "$TRAIN_N0" --exclude-normal

printf '== v2.1 DATA 9/9 N0 Gate0 ==\n'
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$TRAIN_N0" --resize-size 256 \
  --normal-policy none --required-splits train val \
  --certificate-out "$CERT_N0" --certificate-scope training_view \
  --parent-full-certificate "$FULL_CERT"

printf 'REAL_DATA_V21_READY\nFULL_MANIFEST=%s\nTRAIN_N25=%s\nCERT_N25=%s\nTRAIN_N0=%s\nCERT_N0=%s\nFULL_CERT=%s\nTEST_FIREWALL=CLOSED\n' \
  "$FULL_MANIFEST" "$TRAIN_N25" "$CERT_N25" "$TRAIN_N0" "$CERT_N0" "$FULL_CERT"
