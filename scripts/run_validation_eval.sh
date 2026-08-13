#!/usr/bin/env bash
# Validation-only evaluation. Canonical test is intentionally inaccessible here.
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${MANIFEST:-$EXP_ROOT/data/manifest_trainval_with_normal.jsonl}"
ARM_ROOT="${ARM_ROOT:-$EXP_ROOT/arms}"
EVAL_ROOT="${EVAL_ROOT:-$EXP_ROOT/validation_eval}"
mkdir -p "$EVAL_ROOT"

test -f "$MANIFEST" || { echo "MISSING TRAIN/VAL MANIFEST: $MANIFEST" >&2; exit 2; }
if grep -q '"split"[[:space:]]*:[[:space:]]*"test"' "$MANIFEST"; then
  echo "REFUSE: validation evaluator was given a manifest containing canonical test rows" >&2
  exit 3
fi

for name in S0_control S1_oasis_rc_v2 S2_aosk S3_oasis_rc_v2_aosk; do
  ckpt="$ARM_ROOT/$name/student_only.pt"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP $name: $ckpt missing"
    continue
  fi
  out="$EVAL_ROOT/${name}_val.json"
  echo "===== EVAL $name / val ====="
  "$PYTHON" -m oasis_cycle_aosk.evaluate_rc \
    --checkpoint "$ckpt" --manifest "$MANIFEST" --split val \
    --device cuda --out "$out" 2>&1 | tee "$EVAL_ROOT/${name}_val.log"
done

echo "VALIDATION_EVAL_DONE $(date -u +%FT%TZ)"
echo "TEST_FIREWALL=CLOSED"
