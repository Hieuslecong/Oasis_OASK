#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"; DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"; NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"; SEED="${SEED:-1337}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml}"; export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_with_normal.jsonl"; TRAIN_CERT="$DATA_ROOT/gate0_training.json"; FULL_MANIFEST="$DATA_ROOT/manifest_full_with_normal.jsonl"; FULL_CERT="$DATA_ROOT/gate0_full.json"
INIT="$EXP_ROOT/init/student_init_seed${SEED}.pt"; CRITIC_DIR="$EXP_ROOT/critic"; CRITIC="$CRITIC_DIR/critic.pt"; CRITIC_VALIDATION="$CRITIC_DIR/critic_validation.json"; mkdir -p "$EXP_ROOT/init" "$CRITIC_DIR"
if [ "${PREPARE_DATA:-1}" = "1" ]; then DATA_ROOT="$DATA_ROOT" CANONICAL_MANIFEST="$CANONICAL_MANIFEST" NORMAL_ROOT="$NORMAL_ROOT" PYTHON="$PYTHON" EMPTY_CERTIFICATION_CSV="${EMPTY_CERTIFICATION_CSV:-}" "$PACKAGE_ROOT/scripts/prepare_real_data.sh"; fi
for f in "$TRAIN_MANIFEST" "$TRAIN_CERT" "$FULL_MANIFEST" "$FULL_CERT" "$CONFIG"; do test -f "$f" || { echo "MISSING: $f" >&2; exit 2; }; done
echo "== SEED $SEED 1/3 init =="
if [ ! -f "$INIT" ]; then "$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" --seed "$SEED" --student-kind "${STUDENT_KIND:-multiscale}" --student-width "${STUDENT_WIDTH:-16}" --out "$INIT"; fi
echo "== SEED $SEED 2/3 critic =="
if [ "${REUSE_CRITIC:-0}" != "1" ]; then rm -f "$CRITIC" "$CRITIC_VALIDATION"; fi
if [ ! -f "$CRITIC" ]; then "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 --config "$CONFIG" --manifest "$TRAIN_MANIFEST" --gate0-certificate "$TRAIN_CERT" --out "$CRITIC_DIR" --mode critic --normal-fraction "${NORMAL_FRACTION:-0.25}" --normal-critic-weight "${NORMAL_CRITIC_WEIGHT:-1.0}" --critic-epochs "${CRITIC_EPOCHS:-10}" --deterministic; fi
test -f "$CRITIC"; test -f "$CRITIC_VALIDATION"
"$PYTHON" - "$CRITIC_VALIDATION" <<'PY'
import json,sys
from oasis_rc_v2.qualification import critic_gate_failures
m=json.load(open(sys.argv[1])); failed=critic_gate_failures(m); print(json.dumps({"critic_gate":"FAIL" if failed else "PASS","failed":failed,"metrics":m},indent=2))
if failed: raise SystemExit(4)
PY
echo "== SEED $SEED 3/3 S0-S3 validation =="
if [ "${RUN_ARMS:-1}" = "1" ]; then EXP_ROOT="$EXP_ROOT" PYTHON="$PYTHON" CONFIG="$CONFIG" MANIFEST="$TRAIN_MANIFEST" GATE0_CERTIFICATE="$TRAIN_CERT" STUDENT_INIT="$INIT" CRITIC="$CRITIC" NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}" STUDENT_KIND="${STUDENT_KIND:-multiscale}" STUDENT_WIDTH="${STUDENT_WIDTH:-16}" EPOCHS="${EPOCHS:-12}" "$PACKAGE_ROOT/scripts/run_validation_arms.sh"; fi
echo "TRAINING_PIPELINE_READY seed=$SEED"; echo "TEST_FIREWALL=CLOSED"
