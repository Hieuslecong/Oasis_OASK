#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:-}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
SEED="${SEED:-1337}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set NORMAL_FRACTION explicitly: 0.0 for N0 or 0.25 for N25}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml}"
DETERMINISM_MODE="${DETERMINISM_MODE:-best_effort}"
LAMBDA_OASIS="${LAMBDA_OASIS:-0.001}"
LAMBDA_AOSK="${LAMBDA_AOSK:-0.01}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

TRAINING_VARIANT="$($PYTHON - "$NORMAL_FRACTION" <<'PY'
import sys
value=float(sys.argv[1])
if value == 0.0:
    print("N0")
elif value == 0.25:
    print("N25")
else:
    raise SystemExit("official protocol requires NORMAL_FRACTION=0.0 (N0) or 0.25 (N25)")
PY
)"

if [ "$TRAINING_VARIANT" = "N0" ]; then
  TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_n0.jsonl"
  TRAIN_CERT="$DATA_ROOT/gate0_training_n0.json"
else
  test -n "$NORMAL_ROOT" || { echo "N25 requires NORMAL_ROOT" >&2; exit 2; }
  test -d "$NORMAL_ROOT" || { echo "N25 NORMAL_ROOT not found: $NORMAL_ROOT" >&2; exit 2; }
  TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_with_normal.jsonl"
  TRAIN_CERT="$DATA_ROOT/gate0_training.json"
fi

INIT="$EXP_ROOT/init/student_init_seed${SEED}.pt"
CRITIC_DIR="$EXP_ROOT/critic"
CRITIC="$CRITIC_DIR/critic.pt"
CRITIC_VALIDATION="$CRITIC_DIR/critic_validation.json"
mkdir -p "$EXP_ROOT/init" "$CRITIC_DIR"

if [ "${PREPARE_DATA:-1}" = "1" ]; then
  if [ "$TRAINING_VARIANT" = "N0" ]; then
    DATA_ROOT="$DATA_ROOT" \
    CANONICAL_MANIFEST="$CANONICAL_MANIFEST" \
    PYTHON="$PYTHON" \
    EMPTY_CERTIFICATION_CSV="${EMPTY_CERTIFICATION_CSV:-}" \
    "$PACKAGE_ROOT/scripts/prepare_n0_data.sh"
  else
    DATA_ROOT="$DATA_ROOT" \
    CANONICAL_MANIFEST="$CANONICAL_MANIFEST" \
    NORMAL_ROOT="$NORMAL_ROOT" \
    PYTHON="$PYTHON" \
    EMPTY_CERTIFICATION_CSV="${EMPTY_CERTIFICATION_CSV:-}" \
    "$PACKAGE_ROOT/scripts/prepare_real_data.sh"
  fi
fi

for f in "$TRAIN_MANIFEST" "$TRAIN_CERT" "$CONFIG"; do
  test -f "$f" || { echo "MISSING: $f" >&2; exit 2; }
done

"$PYTHON" - "$CONFIG" "$DETERMINISM_MODE" "$NORMAL_FRACTION" <<'PY'
import json, os, sys, yaml, torch
cfg=yaml.safe_load(open(sys.argv[1])); mode=sys.argv[2]; normal=float(sys.argv[3])
if normal not in (0.0, 0.25):
    raise SystemExit("official protocol requires NORMAL_FRACTION=0.0 or 0.25")
if str(cfg.get("device", "cpu")).startswith("cuda") and not torch.cuda.is_available():
    raise SystemExit("CUDA config selected but torch.cuda.is_available() is false")
print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "normal_fraction": normal,
    "determinism_mode": mode,
    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
}, indent=2))
PY

echo "== SEED $SEED 1/3 init =="
if [ ! -f "$INIT" ]; then
  "$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" \
    --seed "$SEED" \
    --student-kind "${STUDENT_KIND:-multiscale}" \
    --student-width "${STUDENT_WIDTH:-16}" \
    --out "$INIT"
fi

echo "== SEED $SEED 2/3 critic ($TRAINING_VARIANT) =="
if [ "${REUSE_CRITIC:-0}" != "1" ]; then
  rm -f "$CRITIC" "$CRITIC_VALIDATION"
fi
if [ ! -f "$CRITIC" ]; then
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" \
    --manifest "$TRAIN_MANIFEST" \
    --gate0-certificate "$TRAIN_CERT" \
    --out "$CRITIC_DIR" \
    --mode critic \
    --normal-fraction "$NORMAL_FRACTION" \
    --normal-critic-weight "${NORMAL_CRITIC_WEIGHT:-1.0}" \
    --critic-epochs "${CRITIC_EPOCHS:-10}" \
    --determinism-mode "$DETERMINISM_MODE"
fi

test -f "$CRITIC"
test -f "$CRITIC_VALIDATION"
"$PYTHON" - "$CRITIC_VALIDATION" <<'PY'
import json,sys
from oasis_rc_v2.qualification import critic_gate_failures
m=json.load(open(sys.argv[1])); failed=critic_gate_failures(m)
print(json.dumps({"critic_gate":"FAIL" if failed else "PASS","failed":failed,"metrics":m},indent=2))
if failed: raise SystemExit(4)
PY

echo "== SEED $SEED 3/3 S0-S3 validation =="
if [ "${RUN_ARMS:-1}" = "1" ]; then
  EXP_ROOT="$EXP_ROOT" \
  PYTHON="$PYTHON" \
  CONFIG="$CONFIG" \
  MANIFEST="$TRAIN_MANIFEST" \
  GATE0_CERTIFICATE="$TRAIN_CERT" \
  STUDENT_INIT="$INIT" \
  CRITIC="$CRITIC" \
  NORMAL_FRACTION="$NORMAL_FRACTION" \
  STUDENT_KIND="${STUDENT_KIND:-multiscale}" \
  STUDENT_WIDTH="${STUDENT_WIDTH:-16}" \
  EPOCHS="${EPOCHS:-12}" \
  LAMBDA_OASIS="$LAMBDA_OASIS" \
  LAMBDA_AOSK="$LAMBDA_AOSK" \
  DETERMINISM_MODE="$DETERMINISM_MODE" \
  "$PACKAGE_ROOT/scripts/run_validation_arms.sh"
fi

echo "TRAINING_PIPELINE_READY seed=$SEED"
echo "TRAINING_VARIANT=$TRAINING_VARIANT"
echo "NORMAL_FRACTION=$NORMAL_FRACTION"
echo "AOSK_VARIANT=oriented-consistency-v1"
echo "DETERMINISM_MODE=$DETERMINISM_MODE"
echo "TEST_FIREWALL=CLOSED"
