#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_three_seeds.sh trainval_manifest.jsonl gate0_training.json [student_kind]}"
GATE0_CERTIFICATE="${2:?Missing Gate 0 training-view certificate}"
STUDENT_KIND="${3:-multiscale}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for SEED in 1337 2027 31415; do
  CONFIG="$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml"
  ROOT="${RUN_ROOT_BASE:-$PACKAGE_ROOT/runs}/canonical_v2_${STUDENT_KIND}_seed${SEED}"
  mkdir -p "$ROOT"
  INIT="$ROOT/student_init_seed${SEED}.pt"
  "$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" --seed "$SEED" --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" --out "$INIT"
  CONFIG="$CONFIG" RUN_ROOT="$ROOT" PYTHON="$PYTHON" STUDENT_WIDTH="$STUDENT_WIDTH" NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}" "$PACKAGE_ROOT/scripts/run_smoke.sh" "$MANIFEST" "$GATE0_CERTIFICATE" "$INIT" "$STUDENT_KIND"
done
echo "Three-seed canonical S0/S1/S2/S3 smoke complete."; echo "TEST_FIREWALL=CLOSED"
