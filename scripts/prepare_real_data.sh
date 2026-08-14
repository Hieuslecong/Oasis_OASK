#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CLEAN_DIR="$DATA_ROOT/cleaned"
BENCH_DIR="$DATA_ROOT/cleaneval_v1"
NORMAL_AUDIT="$DATA_ROOT/normal_audit"
FULL_MANIFEST="$DATA_ROOT/manifest_full_with_normal.jsonl"
TRAIN_N25="$DATA_ROOT/manifest_trainval_with_normal.jsonl"
TRAIN_N0="$DATA_ROOT/manifest_trainval_n0.jsonl"
FULL_CERT="$DATA_ROOT/gate0_full.json"
CERT_N25="$DATA_ROOT/gate0_training.json"
CERT_N0="$DATA_ROOT/gate0_training_n0.json"
NORMAL_EXCLUSIONS="$NORMAL_AUDIT/normal_exclusions.json"
mkdir -p "$CLEAN_DIR" "$BENCH_DIR" "$NORMAL_AUDIT"

echo "== DATA 1/9 leakage + exact-duplicate repair =="
"$PYTHON" "$PACKAGE_ROOT/scripts/clean_manifest.py" \
  --input "$CANONICAL_MANIFEST" \
  --out-dir "$CLEAN_DIR" \
  --resize-size 256

echo "== DATA 2/9 CleanEval fail-closed =="
ARGS=(--input "$CLEAN_DIR/manifest_clean.jsonl" --out-dir "$BENCH_DIR" --resize-size 256)
if [ -n "${EMPTY_CERTIFICATION_CSV:-}" ]; then
  ARGS+=(--certification-csv "$EMPTY_CERTIFICATION_CSV")
fi
"$PYTHON" "$PACKAGE_ROOT/scripts/build_cleaneval_v1.py" "${ARGS[@]}"

echo "== DATA 3/9 normal RGB audit =="
"$PYTHON" "$PACKAGE_ROOT/scripts/audit_normal_rgb_source.py" \
  --normal-root "$NORMAL_ROOT" \
  --cracked-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl" \
  --repair-exclusions-out "$NORMAL_EXCLUSIONS" \
  --out-dir "$NORMAL_AUDIT"

echo "== DATA 4/9 append audited normals =="
"$PYTHON" "$PACKAGE_ROOT/scripts/add_normal_rgb_to_manifest.py" \
  --canonical-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl" \
  --normal-root "$NORMAL_ROOT" \
  --out "$FULL_MANIFEST" \
  --train-ratio 1.0 \
  --exclude-file "$NORMAL_EXCLUSIONS"

echo "== DATA 5/9 full benchmark Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$FULL_MANIFEST" \
  --resize-size 256 \
  --normal-policy train \
  --required-splits train val test \
  --certificate-out "$FULL_CERT" \
  --certificate-scope full_benchmark

echo "== DATA 6/9 N25 training view =="
"$PYTHON" "$PACKAGE_ROOT/scripts/build_training_view.py" \
  --input "$FULL_MANIFEST" \
  --out "$TRAIN_N25"

echo "== DATA 7/9 N25 training-view Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$TRAIN_N25" \
  --resize-size 256 \
  --normal-policy train \
  --required-splits train val \
  --certificate-out "$CERT_N25" \
  --certificate-scope training_view \
  --parent-full-certificate "$FULL_CERT"

echo "== DATA 8/9 N0 crack-only training view =="
"$PYTHON" "$PACKAGE_ROOT/scripts/build_training_view.py" \
  --input "$FULL_MANIFEST" \
  --out "$TRAIN_N0" \
  --exclude-normal

echo "== DATA 9/9 N0 training-view Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$TRAIN_N0" \
  --resize-size 256 \
  --normal-policy none \
  --required-splits train val \
  --certificate-out "$CERT_N0" \
  --certificate-scope training_view \
  --parent-full-certificate "$FULL_CERT"

printf 'REAL_DATA_READY\nFULL_MANIFEST=%s\nTRAIN_N25=%s\nCERT_N25=%s\nTRAIN_N0=%s\nCERT_N0=%s\nFULL_CERT=%s\n' \
  "$FULL_MANIFEST" "$TRAIN_N25" "$CERT_N25" "$TRAIN_N0" "$CERT_N0" "$FULL_CERT"
