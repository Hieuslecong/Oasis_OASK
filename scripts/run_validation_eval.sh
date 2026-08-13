#!/usr/bin/env bash
# PHASE N: evaluate each trained arm on val + test, aggregate normal-aware metrics.
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
PYTHON=/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MANIFEST="$EXP_ROOT/data/derived_manifest_with_normal.jsonl"
ARM_ROOT="$EXP_ROOT/n0_n25"
EVAL_ROOT="$EXP_ROOT/validation_seed1337"
mkdir -p "$EVAL_ROOT"

for name in S0_control S1_oasis_rc_v2 S2_aosk S3_oasis_rc_v2_aosk; do
  ckpt="$ARM_ROOT/$name/student_only.pt"
  if [ ! -f "$ckpt" ]; then
    echo "SKIP $name: $ckpt missing"
    continue
  fi
  for split in val test; do
    out="$EVAL_ROOT/${name}_${split}.json"
    echo "===== EVAL $name / $split ====="
    $PYTHON -m oasis_cycle_aosk.evaluate_rc \
      --checkpoint "$ckpt" --manifest "$MANIFEST" --split "$split" \
      --device cuda --out "$out" 2>&1 | tee "$EVAL_ROOT/${name}_${split}.log"
  done
done
echo "EVAL_DONE $(date -u +%FT%TZ)"
