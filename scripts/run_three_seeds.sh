#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_three_seeds.sh manifest.jsonl [student_kind]}"
STUDENT_KIND="${2:-multiscale}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}"
LAMBDA_OASIS="${LAMBDA_OASIS:-0.001}"
LAMBDA_AOSK="${LAMBDA_AOSK:-0.01}"
EPOCHS="${EPOCHS:-12}"
CRITIC_EPOCHS="${CRITIC_EPOCHS:-12}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for SEED in 1337 2027 31415; do
  CONFIG="$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml"
  RUN_ROOT="${RUN_ROOT_BASE:-$PACKAGE_ROOT/runs}/paired_v2_${STUDENT_KIND}_seed${SEED}"
  mkdir -p "$RUN_ROOT"

  INIT="$RUN_ROOT/student_init_seed${SEED}.pt"
  python "$PACKAGE_ROOT/scripts/create_student_init.py" \
    --seed "$SEED" --student-kind "$STUDENT_KIND" \
    --student-width "$STUDENT_WIDTH" --out "$INIT"

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/critic" --mode critic --critic-epochs "$CRITIC_EPOCHS" \
    --normal-fraction "$NORMAL_FRACTION" --normal-critic-weight 1.0 \
    --pair-weight 0.25 --deterministic

  CRITIC="$RUN_ROOT/critic/critic.pt"

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/S0_control" --mode control \
    --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "$EPOCHS" --normal-fraction "$NORMAL_FRACTION" --deterministic \
    --student-init-checkpoint "$INIT"

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/S1_oasis" --mode connected \
    --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "$EPOCHS" --warmup 4 --ramp-epochs 3 \
    --normal-fraction "$NORMAL_FRACTION" --lambda-oasis "$LAMBDA_OASIS" \
    --deterministic --student-init-checkpoint "$INIT" \
    --critic-checkpoint "$CRITIC"

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/S2_aosk" --mode aosk \
    --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "$EPOCHS" --normal-fraction "$NORMAL_FRACTION" \
    --lambda-aosk "$LAMBDA_AOSK" --deterministic \
    --student-init-checkpoint "$INIT"

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/S3_oasis_aosk" --mode aosk_connected \
    --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "$EPOCHS" --warmup 4 --ramp-epochs 3 \
    --normal-fraction "$NORMAL_FRACTION" \
    --lambda-oasis "$LAMBDA_OASIS" --lambda-aosk "$LAMBDA_AOSK" \
    --deterministic --student-init-checkpoint "$INIT" \
    --critic-checkpoint "$CRITIC"
done

echo "Three-seed S0/S1/S2/S3 validation runs complete. Test has not been evaluated."
