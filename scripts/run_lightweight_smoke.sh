#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_lightweight_smoke.sh manifest.jsonl}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for STUDENT_KIND in mobilenetv3 dsunet fastscnn bisenet; do
  ROOT="${RUN_ROOT_BASE:-$PACKAGE_ROOT/runs}/lightweight_micro_${STUDENT_KIND}"
  mkdir -p "$ROOT"
  INIT="$ROOT/student_init_seed1337.pt"
  python "$PACKAGE_ROOT/scripts/create_student_init.py" \
    --seed 1337 --student-kind "$STUDENT_KIND" --student-width 16 --out "$INIT"
  RUN_ROOT="$ROOT" "$PACKAGE_ROOT/scripts/run_smoke.sh" \
    "$MANIFEST" "$INIT" "$STUDENT_KIND"
done
