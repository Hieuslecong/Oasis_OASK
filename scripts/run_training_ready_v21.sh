#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"
EXP_ROOT="${EXP_ROOT:?set EXP_ROOT}"
DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT}"
LINEAGE_REGEX="${LINEAGE_REGEX:?set LINEAGE_REGEX}"
PYTHON="${PYTHON:-python}"
SEED="${SEED:-1337}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set 0.0 for N0 or 0.25 for N25}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml}"
STUDENT_KIND="${STUDENT_KIND:-mobilenetv3}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
DETERMINISM_MODE="${DETERMINISM_MODE:-strict}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

VARIANT="$($PYTHON - "$NORMAL_FRACTION" <<'PY'
import sys
v=float(sys.argv[1])
if v==0.0: print('N0')
elif v==0.25: print('N25')
else: raise SystemExit('NORMAL_FRACTION must be 0.0 or 0.25')
PY
)"

if [ "${PREPARE_DATA:-1}" = "1" ]; then
  DATA_ROOT="$DATA_ROOT" CANONICAL_MANIFEST="$CANONICAL_MANIFEST" NORMAL_ROOT="$NORMAL_ROOT" \
  LINEAGE_REGEX="$LINEAGE_REGEX" PYTHON="$PYTHON" SEED="$SEED" \
  EMPTY_CERTIFICATION_CSV="${EMPTY_CERTIFICATION_CSV:-}" \
  bash "$PACKAGE_ROOT/scripts/prepare_real_data_v21.sh"
fi

FULL_CERT="$DATA_ROOT/gate0_full_v21.json"
if [ "$VARIANT" = "N25" ]; then
  MANIFEST="$DATA_ROOT/manifest_trainval_n25_v21.jsonl"
  TRAIN_CERT="$DATA_ROOT/gate0_training_n25_v21.json"
else
  MANIFEST="$DATA_ROOT/manifest_trainval_n0_v21.jsonl"
  TRAIN_CERT="$DATA_ROOT/gate0_training_n0_v21.json"
fi
for f in "$CONFIG" "$MANIFEST" "$TRAIN_CERT" "$FULL_CERT"; do
  test -f "$f" || { echo "MISSING: $f" >&2; exit 2; }
done

mkdir -p "$EXP_ROOT/preflight" "$EXP_ROOT/init" "$EXP_ROOT/arms/B0" "$EXP_ROOT/critic" "$EXP_ROOT/diagnostics"

printf '== v2.1-dev2 1/6 real GPU preflight ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/preflight_v21_real_gpu.py" \
  --config "$CONFIG" --manifest "$MANIFEST" \
  --gate0-certificate "$TRAIN_CERT" --full-gate0-certificate "$FULL_CERT" \
  --normal-fraction "$NORMAL_FRACTION" --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
  --critic-width "${CRITIC_WIDTH:-8}" --determinism-mode "$DETERMINISM_MODE" \
  --min-gpu-gib "${MIN_GPU_GIB:-0}" --min-disk-gib "${MIN_DISK_GIB:-1}" \
  --out "$EXP_ROOT/preflight/${VARIANT}_seed${SEED}.json"

INIT="$EXP_ROOT/init/student_init_seed${SEED}_${STUDENT_KIND}.pt"
printf '== v2.1-dev2 2/6 shared initialization ==\n'
if [ -f "$INIT" ] && [ "${REUSE_INIT:-0}" = "1" ]; then
  printf 'REUSE_INIT_REQUESTED=%s\n' "$INIT"
else
  rm -f "$INIT" "$INIT.json"
  "$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" \
    --seed "$SEED" --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" --out "$INIT"
fi

test -f "$INIT"
B0_DIR="$EXP_ROOT/arms/B0"
S0="$B0_DIR/student_only.pt"
printf '== v2.1-dev2 3/6 train canonical B0/S0 development baseline ==\n'
# Default is fail-safe retraining. Reuse is an explicit operational shortcut and
# must only be requested after external provenance validation.
if [ -f "$S0" ] && [ "${REUSE_VALIDATED_B0:-0}" = "1" ]; then
  printf 'REUSE_VALIDATED_B0=%s\n' "$S0"
else
  rm -f "$S0" "$B0_DIR/history.json" "$B0_DIR/validation.json" "$B0_DIR/effective_config.json"
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v21 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --gate0-certificate "$TRAIN_CERT" --full-gate0-certificate "$FULL_CERT" \
    --out "$B0_DIR" --mode control --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "${EPOCHS:-12}" --warmup "${WARMUP:-4}" --ramp-epochs "${RAMP:-3}" \
    --normal-fraction "$NORMAL_FRACTION" --determinism-mode "$DETERMINISM_MODE" \
    --student-init-checkpoint "$INIT" 2>&1 | tee "$B0_DIR/train.log"
fi

test -f "$S0"
CRITIC="$EXP_ROOT/critic/critic.pt"
printf '== v2.1-dev2 4/6 train + qualify critic ==\n'
if [ -f "$CRITIC" ] && [ "${REUSE_VALIDATED_CRITIC:-0}" = "1" ]; then
  printf 'REUSE_VALIDATED_CRITIC=%s\n' "$CRITIC"
else
  rm -f "$CRITIC" "$EXP_ROOT/critic/critic_qualification_v21.json"
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v21 \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --gate0-certificate "$TRAIN_CERT" --full-gate0-certificate "$FULL_CERT" \
    --out "$EXP_ROOT/critic" --mode critic --normal-fraction "$NORMAL_FRACTION" \
    --critic-width "${CRITIC_WIDTH:-8}" --critic-epochs "${CRITIC_EPOCHS:-10}" \
    --endpoint-weight "${ENDPOINT_WEIGHT:-1.0}" --endpoint-anchor-weight "${ENDPOINT_ANCHOR_WEIGHT:-0.25}" \
    --endpoint-margin "${ENDPOINT_MARGIN:-0.05}" --path-weight "${PATH_WEIGHT:-1.0}" --path-margin "${PATH_MARGIN:-0.02}" \
    --determinism-mode "$DETERMINISM_MODE" 2>&1 | tee "$EXP_ROOT/critic/train.log"
fi

test -f "$CRITIC"
test -f "$EXP_ROOT/critic/critic_qualification_v21.json"
printf '== v2.1-dev2 5/6 trained-B0 RC gradient/energy gate ==\n'
"$PYTHON" "$PACKAGE_ROOT/scripts/diagnose_v21_s0.py" \
  --s0-checkpoint "$S0" --critic-checkpoint "$CRITIC" --manifest "$MANIFEST" \
  --full-gate0-certificate "$FULL_CERT" --device "${DIAGNOSTIC_DEVICE:-cuda}" \
  --lambda-oasis "${LAMBDA_OASIS:-0.001}" --min-s0-epochs "${MIN_S0_EPOCHS:-${EPOCHS:-12}}" \
  --out "$EXP_ROOT/diagnostics/s0_rc_${VARIANT}_seed${SEED}.json" --require-pass

printf '== v2.1-dev2 6/6 B0/B1/B2/S1/S2/S3 development arms ==\n'
EXP_ROOT="$EXP_ROOT" PYTHON="$PYTHON" CONFIG="$CONFIG" MANIFEST="$MANIFEST" \
GATE0_CERTIFICATE="$TRAIN_CERT" FULL_GATE0_CERTIFICATE="$FULL_CERT" STUDENT_INIT="$INIT" CRITIC="$CRITIC" \
NORMAL_FRACTION="$NORMAL_FRACTION" STUDENT_KIND="$STUDENT_KIND" STUDENT_WIDTH="$STUDENT_WIDTH" \
EPOCHS="${EPOCHS:-12}" WARMUP="${WARMUP:-4}" RAMP="${RAMP:-3}" DETERMINISM_MODE="$DETERMINISM_MODE" \
B0_DIR="$B0_DIR" REUSE_VALIDATED_B0=1 \
LAMBDA_OASIS="${LAMBDA_OASIS:-0.001}" LAMBDA_AOSK="${LAMBDA_AOSK:-0.01}" \
LAMBDA_CLDICE="${LAMBDA_CLDICE:-0.1}" LAMBDA_FROZEN_PAIR="${LAMBDA_FROZEN_PAIR:-${LAMBDA_ADVERSARIAL:-0.001}}" \
bash "$PACKAGE_ROOT/scripts/run_validation_arms_v21.sh"

printf 'REAL_DATA_V21_DEV2_DEVELOPMENT_PIPELINE=PASS\nVARIANT=%s\nSEED=%s\nSTUDENT_KIND=%s\nCANONICAL_B0=%s\nTEST_FIREWALL=CLOSED\n' \
  "$VARIANT" "$SEED" "$STUDENT_KIND" "$S0"
