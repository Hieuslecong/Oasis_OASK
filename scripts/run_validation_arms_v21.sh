#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

CONFIG="${CONFIG:?set CONFIG}"
MANIFEST="${MANIFEST:?set MANIFEST}"
GATE0_CERTIFICATE="${GATE0_CERTIFICATE:?set GATE0_CERTIFICATE}"
FULL_GATE0_CERTIFICATE="${FULL_GATE0_CERTIFICATE:?set FULL_GATE0_CERTIFICATE}"
STUDENT_INIT="${STUDENT_INIT:?set STUDENT_INIT}"
CRITIC="${CRITIC:?set CRITIC}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set NORMAL_FRACTION}"
STUDENT_KIND="${STUDENT_KIND:-mobilenetv3}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
EPOCHS="${EPOCHS:-12}"
WARMUP="${WARMUP:-4}"
RAMP="${RAMP:-3}"
DETERMINISM_MODE="${DETERMINISM_MODE:-strict}"
ARM_ROOT="${ARM_ROOT:-$EXP_ROOT/arms}"
mkdir -p "$ARM_ROOT"

for f in "$CONFIG" "$MANIFEST" "$GATE0_CERTIFICATE" "$FULL_GATE0_CERTIFICATE" "$STUDENT_INIT" "$CRITIC"; do
  test -f "$f" || { echo "MISSING: $f" >&2; exit 2; }
done

run_arm() {
  local name="$1" mode="$2"; shift 2
  local out="$ARM_ROOT/$name"
  mkdir -p "$out"
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v21 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --gate0-certificate "$GATE0_CERTIFICATE" --full-gate0-certificate "$FULL_GATE0_CERTIFICATE" \
    --out "$out" --mode "$mode" --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "$EPOCHS" --warmup "$WARMUP" --ramp-epochs "$RAMP" \
    --normal-fraction "$NORMAL_FRACTION" --determinism-mode "$DETERMINISM_MODE" \
    --student-init-checkpoint "$STUDENT_INIT" "$@" 2>&1 | tee "$out/train.log"
}

# B0 may already exist because run_training_ready_v21.sh trains it first for the
# S0-manifold diagnostic. Reuse is allowed only when that parent launcher has
# explicitly provided the same checkpoint path; standalone runs retrain B0.
B0_DIR="${B0_DIR:-$ARM_ROOT/B0}"
if [ "${REUSE_VALIDATED_B0:-0}" = "1" ] && [ -f "$B0_DIR/student_only.pt" ]; then
  printf 'REUSE_VALIDATED_B0=%s\n' "$B0_DIR/student_only.pt"
else
  run_arm B0 control
fi
run_arm B1_cldice cldice --lambda-cldice "${LAMBDA_CLDICE:-0.1}"
# Internal mode name 'adversarial' is retained for checkpoint compatibility;
# scientifically this arm is a frozen pretrained pair-critic ablation.
run_arm B2_frozen_pair adversarial --lambda-adversarial "${LAMBDA_FROZEN_PAIR:-${LAMBDA_ADVERSARIAL:-0.001}}" --critic-checkpoint "$CRITIC"
run_arm S1_rc connected --lambda-oasis "${LAMBDA_OASIS:-0.001}" --critic-checkpoint "$CRITIC"
run_arm S2_aosk aosk --lambda-aosk "${LAMBDA_AOSK:-0.01}"
run_arm S3_rc_aosk aosk_connected --lambda-oasis "${LAMBDA_OASIS:-0.001}" --lambda-aosk "${LAMBDA_AOSK:-0.01}" --critic-checkpoint "$CRITIC"

printf 'V21_DEVELOPMENT_ARMS_DONE\nSTUDENT_KIND=%s\nNORMAL_FRACTION=%s\nB2_SEMANTICS=frozen-pretrained-pair-critic\nTEST_FIREWALL=CLOSED\n' "$STUDENT_KIND" "$NORMAL_FRACTION"
