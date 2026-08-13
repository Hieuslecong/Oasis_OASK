#!/usr/bin/env bash
# End-to-end fail-closed preparation + seed-1337 validation training.
# Required: EXP_ROOT, CANONICAL_MANIFEST, NORMAL_ROOT
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed1337.yaml}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CLEAN_DIR="$EXP_ROOT/data/cleaned_fixed"
BENCH_DIR="$EXP_ROOT/data/cleaneval_v1_fixed"
TRAIN_MANIFEST="$EXP_ROOT/data/derived_manifest_with_normal.jsonl"
INIT="$EXP_ROOT/init/student_init_seed1337.pt"
CRITIC_DIR="$EXP_ROOT/critic"
CRITIC="$CRITIC_DIR/critic.pt"
mkdir -p "$CLEAN_DIR" "$BENCH_DIR" "$EXP_ROOT/init" "$CRITIC_DIR"

echo "== 1/6 leakage repair (test > val > train) =="
"$PYTHON" "$PACKAGE_ROOT/scripts/clean_manifest.py" \
  --input "$CANONICAL_MANIFEST" --out-dir "$CLEAN_DIR" --resize-size 256

echo "== 2/6 fail-closed CleanEval build =="
BUILD_ARGS=(--input "$CLEAN_DIR/manifest_clean.jsonl" --out-dir "$BENCH_DIR" --resize-size 256)
if [ -n "${EMPTY_CERTIFICATION_CSV:-}" ]; then
  BUILD_ARGS+=(--certification-csv "$EMPTY_CERTIFICATION_CSV")
fi
"$PYTHON" "$PACKAGE_ROOT/scripts/build_cleaneval_v1.py" "${BUILD_ARGS[@]}"

echo "== 3/6 append true-normal RGB =="
NORMAL_ARGS=(
  --canonical-manifest "$BENCH_DIR/manifest_cleaneval_v1_full.jsonl"
  --normal-root "$NORMAL_ROOT"
  --out "$TRAIN_MANIFEST"
  --train-ratio 1.0
)
if [ -n "${NORMAL_EXCLUDE_FILE:-}" ]; then
  NORMAL_ARGS+=(--exclude-file "$NORMAL_EXCLUDE_FILE")
fi
"$PYTHON" "$PACKAGE_ROOT/scripts/add_normal_rgb_to_manifest.py" "${NORMAL_ARGS[@]}"

echo "== 4/6 authoritative Gate 0 =="
"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$TRAIN_MANIFEST" --resize-size 256 --normal-policy train

echo "== 5/6 canonical init + critic qualification =="
if [ ! -f "$INIT" ]; then
  "$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" \
    --seed 1337 --student-kind multiscale --student-width 16 --out "$INIT"
fi

if [ "${REUSE_CRITIC:-0}" != "1" ]; then
  rm -f "$CRITIC"
fi
if [ ! -f "$CRITIC" ]; then
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$TRAIN_MANIFEST" \
    --out "$CRITIC_DIR" --mode critic \
    --normal-fraction "${NORMAL_FRACTION:-0.25}" \
    --normal-critic-weight "${NORMAL_CRITIC_WEIGHT:-1.0}" \
    --critic-epochs "${CRITIC_EPOCHS:-10}" \
    --deterministic
fi

test -f "$CRITIC"

echo "== 6/6 validation-only S0/S1/S2/S3 =="
if [ "${RUN_ARMS:-1}" = "1" ]; then
  EXP_ROOT="$EXP_ROOT" PYTHON="$PYTHON" CONFIG="$CONFIG" \
  MANIFEST="$TRAIN_MANIFEST" STUDENT_INIT="$INIT" CRITIC="$CRITIC" \
  NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}" \
  "$PACKAGE_ROOT/scripts/run_validation_arms.sh"
else
  echo "RUN_ARMS=0: preparation/critic complete; student arms not started."
fi

echo "TRAINING_PIPELINE_READY"
echo "TEST_FIREWALL=CLOSED"
