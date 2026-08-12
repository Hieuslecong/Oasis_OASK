#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_three_seeds.sh /absolute/path/to/source_disjoint_manifest.jsonl [student_kind]}"
STUDENT_KIND="${2:-mobilenetv3}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

python -m oasis_cycle_aosk.audit \
  --manifest "$MANIFEST" --test-split test --require-source-disjoint

for SEED in 1337 2027 31415; do
  CONFIG="$PACKAGE_ROOT/configs/debug_cpu_128.yaml"
  if [[ "$SEED" == "2027" ]]; then CONFIG="$PACKAGE_ROOT/configs/debug_cpu_128_seed2027.yaml"; fi
  if [[ "$SEED" == "31415" ]]; then CONFIG="$PACKAGE_ROOT/configs/debug_cpu_128_seed31415.yaml"; fi
  RUN_ROOT="$PACKAGE_ROOT/runs/full_v2_${STUDENT_KIND}_seed${SEED}"

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/critic" --mode critic --critic-epochs 12 \
    --pair-weight 0.25 --test-split test

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/control" --mode control \
    --student-kind "$STUDENT_KIND" --epochs 12 --test-split test

  python -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --out "$RUN_ROOT/connected" --mode connected \
    --student-kind "$STUDENT_KIND" --epochs 12 --warmup 4 --ramp-epochs 3 \
    --lambda-oasis 0.003 --student-pair-weight 0.25 \
    --corrupted-rank-weight 1.0 \
    --critic-checkpoint "$RUN_ROOT/critic/critic.pt" --test-split test
done
