#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_smoke.sh /absolute/path/to/manifest.jsonl [student_kind]}"
STUDENT_KIND="${2:-mobilenetv3}"
CONFIG="$PACKAGE_ROOT/configs/debug_cpu_128.yaml"
RUN_ROOT="$PACKAGE_ROOT/runs/smoke_v2_${STUDENT_KIND}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m oasis_cycle_aosk.audit --manifest "$MANIFEST" --test-split test_debug

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/critic" --mode critic --critic-epochs 3 \
  --pair-weight 0.25 --test-split test_debug

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/control" --mode control \
  --student-kind "$STUDENT_KIND" --epochs 3 --test-split test_debug

python -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --out "$RUN_ROOT/connected" --mode connected \
  --student-kind "$STUDENT_KIND" --epochs 3 --warmup 1 --ramp-epochs 2 \
  --lambda-oasis 0.003 --student-pair-weight 0.25 \
  --corrupted-rank-weight 1.0 \
  --critic-checkpoint "$RUN_ROOT/critic/critic.pt" \
  --test-split test_debug

echo "OASIS-RC-v2 smoke artifacts: $RUN_ROOT"
