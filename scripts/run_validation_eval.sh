#!/usr/bin/env bash
# Validation-only evaluation. Canonical test is intentionally inaccessible here.
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="${MANIFEST:?set MANIFEST explicitly to the certified train/val view}"
ARM_ROOT="${ARM_ROOT:-$EXP_ROOT/arms}"
EVAL_ROOT="${EVAL_ROOT:-$EXP_ROOT/validation_eval}"
DEVICE="${DEVICE:-cuda}"
mkdir -p "$EVAL_ROOT"

test -f "$MANIFEST" || { echo "MISSING TRAIN/VAL MANIFEST: $MANIFEST" >&2; exit 2; }
if grep -q '"split"[[:space:]]*:[[:space:]]*"test"' "$MANIFEST"; then
  echo "REFUSE: validation evaluator was given a manifest containing canonical test rows" >&2
  exit 3
fi

arms=(
  S0_control
  S1_oasis_rc_v2
  S2_aosk
  S3_oasis_rc_v2_aosk
)

for name in "${arms[@]}"; do
  ckpt="$ARM_ROOT/$name/student_only.pt"
  test -f "$ckpt" || { echo "FAIL: official validation arm missing $ckpt" >&2; exit 4; }
  out="$EVAL_ROOT/${name}_val.json"
  echo "===== EVAL $name / val ====="
  "$PYTHON" -m oasis_cycle_aosk.evaluate_rc \
    --checkpoint "$ckpt" \
    --manifest "$MANIFEST" \
    --split val \
    --device "$DEVICE" \
    --out "$out" 2>&1 | tee "$EVAL_ROOT/${name}_val.log"
done

echo "VALIDATION_EVAL_DONE $(date -u +%FT%TZ)"
echo "AOSK_VARIANT=oriented-consistency-v1"
echo "TEST_FIREWALL=CLOSED"
