#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CLEAN_DIR="$DATA_ROOT/cleaned"; BENCH_DIR="$DATA_ROOT/cleaneval_v1"; NORMAL_AUDIT="$DATA_ROOT/normal_audit"
FULL_MANIFEST="$DATA_ROOT/manifest_full_with_normal.jsonl"; TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_with_normal.jsonl"
FULL_CERT="$DATA_ROOT/gate0_full.json"; TRAIN_CERT="$DATA_ROOT/gate0_training.json"; NORMAL_EXCLUSIONS="$NORMAL_AUDIT/normal_exclusions.json"
mkdir -p "$CLEAN_DIR" "$BENCH_DIR" "$NORMAL_AUDIT"
echo "== DATA 1/7 leakage repair =="
"$PYTHON" "$PACKAGE_ROOT/scripts/clean_manifest.py" --input "$CANONICAL_MANIFEST" --out-dir "$CLEAN_DIR" --resize-size 256
echo "== DATA 2/7 CleanEval =="
ARGS=(--input "$CLEAN_DIR/manifest_clean.jsonl" --out-dir "$BENCH_DIR" --resize-size 256)
if [ -n "${EMPTY_CERTIFICATION_CSV:-}" ]; then ARGS+=(--certification-csv "$EMPTY_CERTIFICATION_CSV"); fi
"$PYTHON" "$PACKAGE_ROOT/scripts/build_cleaneval_v1.py" "${ARGS[@]}"
echo "== DATA 3/7 normal RGB audit =="
"$PYTHON" "$PACKAGE_ROOT/scripts/audit_normal_rgb_source.py" --normal-root "$NORMAL_ROOT" --cracked-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl" --repair-exclusions-out "$NORMAL_EXCLUSIONS" --out-dir "$NORMAL_AUDIT"
echo "== DATA 4/7 append audited normals =="
"$PYTHON" "$PACKAGE_ROOT/scripts/add_normal_rgb_to_manifest.py" --canonical-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl" --normal-root "$NORMAL_ROOT" --out "$FULL_MANIFEST" --train-ratio 1.0 --exclude-file "$NORMAL_EXCLUSIONS"
echo "== DATA 5/7 full Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit --manifest "$FULL_MANIFEST" --resize-size 256 --normal-policy train --required-splits train val test --certificate-out "$FULL_CERT" --certificate-scope full_benchmark
echo "== DATA 6/7 training view =="
"$PYTHON" "$PACKAGE_ROOT/scripts/build_training_view.py" --input "$FULL_MANIFEST" --out "$TRAIN_MANIFEST"
echo "== DATA 7/7 training-view Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit --manifest "$TRAIN_MANIFEST" --resize-size 256 --normal-policy train --required-splits train val --certificate-out "$TRAIN_CERT" --certificate-scope training_view
printf 'REAL_DATA_READY\nFULL_MANIFEST=%s\nTRAIN_MANIFEST=%s\nFULL_CERT=%s\nTRAIN_CERT=%s\n' "$FULL_MANIFEST" "$TRAIN_MANIFEST" "$FULL_CERT" "$TRAIN_CERT"
