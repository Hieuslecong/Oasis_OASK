#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CLEAN_DIR="$DATA_ROOT/cleaned_n0"
BENCH_DIR="$DATA_ROOT/cleaneval_n0"
FULL_MANIFEST="$BENCH_DIR/manifest_cleaneval_v1_full.jsonl"
TRAIN_N0="$DATA_ROOT/manifest_trainval_n0.jsonl"
FULL_CERT_N0="$DATA_ROOT/gate0_full_n0.json"
CERT_N0="$DATA_ROOT/gate0_training_n0.json"
mkdir -p "$CLEAN_DIR" "$BENCH_DIR"

echo "== N0 DATA 1/5 leakage + exact-duplicate repair =="
"$PYTHON" "$PACKAGE_ROOT/scripts/clean_manifest.py" \
  --input "$CANONICAL_MANIFEST" \
  --out-dir "$CLEAN_DIR" \
  --resize-size 256

echo "== N0 DATA 2/5 CleanEval fail-closed =="
ARGS=(--input "$CLEAN_DIR/manifest_clean.jsonl" --out-dir "$BENCH_DIR" --resize-size 256)
if [ -n "${EMPTY_CERTIFICATION_CSV:-}" ]; then
  ARGS+=(--certification-csv "$EMPTY_CERTIFICATION_CSV")
fi
"$PYTHON" "$PACKAGE_ROOT/scripts/build_cleaneval_v1.py" "${ARGS[@]}"

echo "== N0 DATA 3/5 crack-only full benchmark Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$FULL_MANIFEST" \
  --resize-size 256 \
  --normal-policy none \
  --required-splits train val test \
  --certificate-out "$FULL_CERT_N0" \
  --certificate-scope full_benchmark

echo "== N0 DATA 4/5 crack-only train/val view =="
"$PYTHON" "$PACKAGE_ROOT/scripts/build_training_view.py" \
  --input "$FULL_MANIFEST" \
  --out "$TRAIN_N0" \
  --exclude-normal

echo "== N0 DATA 5/5 crack-only training-view Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$TRAIN_N0" \
  --resize-size 256 \
  --normal-policy none \
  --required-splits train val \
  --certificate-out "$CERT_N0" \
  --certificate-scope training_view \
  --parent-full-certificate "$FULL_CERT_N0"

printf 'N0_DATA_READY\nTRAIN_N0=%s\nCERT_N0=%s\nFULL_CERT_N0=%s\n' \
  "$TRAIN_N0" "$CERT_N0" "$FULL_CERT_N0"
