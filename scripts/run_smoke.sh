#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_smoke.sh manifest.jsonl student_init.pt [student_kind]}"
STUDENT_INIT="${2:?Missing canonical student init checkpoint}"
STUDENT_KIND="${3:-multiscale}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed1337.yaml}"
NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}"
RUN_ROOT="${RUN_ROOT:-$PACKAGE_ROOT/runs/four_arm_micro_${STUDENT_KIND}}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

IMAGE_SIZE="$(python - "$CONFIG" <<'PY'
import sys, yaml
print(int(yaml.safe_load(open(sys.argv[1]))['image_size']))
PY
)"

python -m oasis_cycle_aosk.audit \
  --manifest "$MANIFEST" --resize-size "$IMAGE_SIZE" \
  --normal-policy train

# Train the critic ONCE. S1 and S3 must load this exact checkpoint.
python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/critic" --mode critic --critic-epochs 2 \
  --normal-fraction "$NORMAL_FRACTION" --normal-critic-weight 1.0 \
  --pair-weight 0.25 --deterministic

CRITIC="$RUN_ROOT/critic/critic.pt"

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/S0_control" --mode control \
  --student-kind "$STUDENT_KIND" --epochs 2 \
  --normal-fraction "$NORMAL_FRACTION" --deterministic \
  --student-init-checkpoint "$STUDENT_INIT"

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/S2_aosk" --mode aosk \
  --student-kind "$STUDENT_KIND" --epochs 2 \
  --normal-fraction "$NORMAL_FRACTION" --lambda-aosk 0.01 --deterministic \
  --student-init-checkpoint "$STUDENT_INIT"

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/S1_oasis" --mode connected \
  --student-kind "$STUDENT_KIND" --epochs 2 --warmup 0 --ramp-epochs 1 \
  --normal-fraction "$NORMAL_FRACTION" --lambda-oasis 0.001 --deterministic \
  --student-init-checkpoint "$STUDENT_INIT" \
  --critic-checkpoint "$CRITIC"

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/S3_oasis_aosk" --mode aosk_connected \
  --student-kind "$STUDENT_KIND" --epochs 2 --warmup 0 --ramp-epochs 1 \
  --normal-fraction "$NORMAL_FRACTION" --lambda-oasis 0.001 --lambda-aosk 0.01 \
  --deterministic --student-init-checkpoint "$STUDENT_INIT" \
  --critic-checkpoint "$CRITIC"

echo "Four-arm validation-only micro-smoke artifacts: $RUN_ROOT"
