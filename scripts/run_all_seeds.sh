#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:?set BASE_EXP_ROOT}"
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:-}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set NORMAL_FRACTION explicitly: 0.0 for N0 or 0.25 for N25}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
DATA_ROOT="${DATA_ROOT:-$BASE_EXP_ROOT/data}"
DETERMINISM_MODE="${DETERMINISM_MODE:-best_effort}"
LAMBDA_OASIS="${LAMBDA_OASIS:-0.001}"
LAMBDA_AOSK="${LAMBDA_AOSK:-0.01}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PROTOCOL="$($PYTHON - "$NORMAL_FRACTION" <<'PY'
import sys
value=float(sys.argv[1])
if value not in (0.0, 0.25):
    raise SystemExit("official 3-seed protocol requires NORMAL_FRACTION=0.0 (N0) or 0.25 (N25)")
print("N0" if value == 0.0 else "N25")
PY
)"
if [ "$PROTOCOL" = "N25" ]; then
  test -n "$NORMAL_ROOT" || { echo "N25 requires NORMAL_ROOT" >&2; exit 2; }
  test -d "$NORMAL_ROOT" || { echo "N25 NORMAL_ROOT not found: $NORMAL_ROOT" >&2; exit 2; }
fi
echo "THREE_SEED_PROTOCOL=$PROTOCOL"

first=1
for seed in 1337 2027 31415; do
  prepare=0
  if [ "$first" = "1" ]; then
    prepare=1
    first=0
  fi
  EXP_ROOT="$BASE_EXP_ROOT/seed_$seed" \
  DATA_ROOT="$DATA_ROOT" \
  CANONICAL_MANIFEST="$CANONICAL_MANIFEST" \
  NORMAL_ROOT="$NORMAL_ROOT" \
  PYTHON="$PYTHON" \
  SEED="$seed" \
  PREPARE_DATA="$prepare" \
  EMPTY_CERTIFICATION_CSV="${EMPTY_CERTIFICATION_CSV:-}" \
  NORMAL_FRACTION="$NORMAL_FRACTION" \
  CRITIC_EPOCHS="${CRITIC_EPOCHS:-10}" \
  EPOCHS="${EPOCHS:-12}" \
  LAMBDA_OASIS="$LAMBDA_OASIS" \
  LAMBDA_AOSK="$LAMBDA_AOSK" \
  RUN_ARMS="${RUN_ARMS:-1}" \
  DETERMINISM_MODE="$DETERMINISM_MODE" \
  "$PACKAGE_ROOT/scripts/run_training_ready.sh"
done

echo "ALL_3_SEEDS_VALIDATION_DONE"
echo "TRAINING_VARIANT=$PROTOCOL"
echo "NORMAL_FRACTION=$NORMAL_FRACTION"
echo "LAMBDA_OASIS=$LAMBDA_OASIS"
echo "LAMBDA_AOSK=$LAMBDA_AOSK"
echo "AOSK_VARIANT=oriented-consistency-v1"
echo "DETERMINISM_MODE=$DETERMINISM_MODE"
echo "TEST_FIREWALL=CLOSED"
