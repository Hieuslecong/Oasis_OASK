#!/usr/bin/env bash
# Four-arm validation-only training on one GPU.
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT before calling}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed1337.yaml}"
MANIFEST="${MANIFEST:-$EXP_ROOT/data/derived_manifest_with_normal.jsonl}"
STUDENT_INIT="${STUDENT_INIT:-$EXP_ROOT/init/student_init_seed1337.pt}"
CRITIC="${CRITIC:-$EXP_ROOT/critic/critic.pt}"
NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}"
STUDENT_KIND="${STUDENT_KIND:-multiscale}"
EPOCHS="${EPOCHS:-12}"
WARMUP="${WARMUP:-4}"
RAMP="${RAMP:-3}"
ARM_ROOT="${ARM_ROOT:-$EXP_ROOT/n0_n25}"

for f in "$MANIFEST" "$STUDENT_INIT" "$CRITIC"; do
  if [ ! -f "$f" ]; then
    echo "MISSING REQUIRED FILE: $f" >&2
    exit 2
  fi
done

"$PYTHON" -m oasis_cycle_aosk.audit \
  --manifest "$MANIFEST" --resize-size 256 --normal-policy train

mkdir -p "$ARM_ROOT"

run_arm () {
  local name="$1"; local mode="$2"; local extra="$3"
  local out="$ARM_ROOT/$name"
  mkdir -p "$out"
  echo "===== START $name (mode=$mode) $(date -u +%FT%TZ) ====="
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$out" --mode "$mode" \
    --student-kind "$STUDENT_KIND" --epochs "$EPOCHS" \
    --warmup "$WARMUP" --ramp-epochs "$RAMP" \
    --normal-fraction "$NORMAL_FRACTION" \
    --deterministic --student-init-checkpoint "$STUDENT_INIT" \
    $extra \
    2>&1 | tee "$out/train.log"
  echo "===== END $name $(date -u +%FT%TZ) ====="
}

run_arm S0_control control ""
run_arm S2_aosk aosk "--lambda-aosk 0.01"
run_arm S1_oasis_rc_v2 connected "--lambda-oasis 0.001 --critic-checkpoint $CRITIC"
run_arm S3_oasis_rc_v2_aosk aosk_connected "--lambda-oasis 0.001 --lambda-aosk 0.01 --critic-checkpoint $CRITIC"

echo "ALL_VALIDATION_ARMS_DONE $(date -u +%FT%TZ)"
echo "TEST_FIREWALL=CLOSED"
