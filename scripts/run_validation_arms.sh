#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

CONFIG="${CONFIG:?set CONFIG}"
MANIFEST="${MANIFEST:?set MANIFEST}"
GATE0_CERTIFICATE="${GATE0_CERTIFICATE:?set GATE0_CERTIFICATE}"
STUDENT_INIT="${STUDENT_INIT:?set STUDENT_INIT}"
CRITIC="${CRITIC:?set CRITIC}"
NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}"
STUDENT_KIND="${STUDENT_KIND:-multiscale}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
EPOCHS="${EPOCHS:-12}"
WARMUP="${WARMUP:-4}"
RAMP="${RAMP:-3}"
DETERMINISM_MODE="${DETERMINISM_MODE:-best_effort}"
ARM_ROOT="${ARM_ROOT:-$EXP_ROOT/arms}"

for f in "$MANIFEST" "$GATE0_CERTIFICATE" "$STUDENT_INIT" "$CRITIC"; do
  test -f "$f" || { echo "MISSING: $f" >&2; exit 2; }
done
mkdir -p "$ARM_ROOT"

run_arm() {
  local name="$1"
  local mode="$2"
  shift 2
  local out="$ARM_ROOT/$name"
  mkdir -p "$out"
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" \
    --manifest "$MANIFEST" \
    --gate0-certificate "$GATE0_CERTIFICATE" \
    --out "$out" \
    --mode "$mode" \
    --student-kind "$STUDENT_KIND" \
    --student-width "$STUDENT_WIDTH" \
    --epochs "$EPOCHS" \
    --warmup "$WARMUP" \
    --ramp-epochs "$RAMP" \
    --normal-fraction "$NORMAL_FRACTION" \
    --determinism-mode "$DETERMINISM_MODE" \
    --student-init-checkpoint "$STUDENT_INIT" \
    "$@" 2>&1 | tee "$out/train.log"
}

run_arm S0_control control
run_arm S2_aosk_oriented aosk --lambda-aosk 0.01
run_arm S1_oasis_rc_v2 connected --lambda-oasis 0.001 --critic-checkpoint "$CRITIC"
run_arm S3_oasis_rc_v2_aosk_oriented aosk_connected --lambda-oasis 0.001 --lambda-aosk 0.01 --critic-checkpoint "$CRITIC"

echo "ALL_VALIDATION_ARMS_DONE"
echo "AOSK_VARIANT=oriented-consistency-v1"
echo "TEST_FIREWALL=CLOSED"
