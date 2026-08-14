#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_lightweight_smoke.sh trainval_manifest.jsonl gate0_training.json}"
GATE0_CERTIFICATE="${2:?Missing Gate 0 training-view certificate}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set NORMAL_FRACTION explicitly: 0.0 or 0.25}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for STUDENT_KIND in mobilenetv3 dsunet fastscnn bisenet; do
  ROOT="${RUN_ROOT_BASE:-$PACKAGE_ROOT/runs}/lightweight_micro_${STUDENT_KIND}"
  mkdir -p "$ROOT"
  INIT="$ROOT/student_init_seed1337.pt"
  "$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" --seed 1337 --student-kind "$STUDENT_KIND" --student-width 16 --out "$INIT"
  RUN_ROOT="$ROOT" PYTHON="$PYTHON" NORMAL_FRACTION="$NORMAL_FRACTION" \
    "$PACKAGE_ROOT/scripts/run_smoke.sh" "$MANIFEST" "$GATE0_CERTIFICATE" "$INIT" "$STUDENT_KIND"
done
