#!/usr/bin/env bash
# PHASE M: four-arm deterministic training on one GPU (fairness: single device, shared init+manifest).
# Executive intent: instantiate S0/S1/S2/S3, train to 12 student epochs each, record optimizer
# step counts (the accountability metric from the brief).
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT before calling}"
PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="$PACKAGE_ROOT/configs/canonical_gpu_256_seed1337.yaml"
MANIFEST="$EXP_ROOT/data/derived_manifest_with_normal.jsonl"
STUDENT_INIT="$EXP_ROOT/init/student_init_seed1337.pt"
CRITIC="$EXP_ROOT/critic/critic.pt"
NORMAL_FRACTION=0.25
STUDENT_KIND=multiscale
EPOCHS=12
WARMUP=4
RAMP=3
ARM_ROOT="$EXP_ROOT/n0_n25"

mkdir -p "$ARM_ROOT"

run_arm () {
  local name="$1"; local mode="$2"; local extra="$3"
  local out="$ARM_ROOT/$name"
  mkdir -p "$out"
  echo "===== START $name (mode=$mode) $(date -u +%FT%TZ) ====="
  $PYTHON -m oasis_cycle_aosk.train_oasis_rc_v2 \
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

# S0 control: baseline, no auxiliary losses.
run_arm S0_control control ""

# S2 aosk: AOSK consistency only.
run_arm S2_aosk aosk "--lambda-aosk 0.01"

# S1 connected: OASIS-RC only, loads shared critic.
run_arm S1_oasis_rc_v2 connected "--lambda-oasis 0.001 --critic-checkpoint $CRITIC"

# S3 aosk_connected: both, loads shared critic.
run_arm S3_oasis_rc_v2_aosk aosk_connected "--lambda-oasis 0.001 --lambda-aosk 0.01 --critic-checkpoint $CRITIC"

echo "ALL_ARMS_DONE $(date -u +%FT%TZ)"
