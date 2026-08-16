#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_smoke.sh trainval_manifest.jsonl gate0_training.json student_init.pt student_kind gate0_full.json}"
GATE0_CERTIFICATE="${2:?Missing Gate 0 training-view certificate}"
STUDENT_INIT="${3:?Missing canonical student init checkpoint}"
STUDENT_KIND="${4:-multiscale}"
FULL_GATE0_CERTIFICATE="${5:?Missing full-benchmark Gate 0 certificate}"
STUDENT_WIDTH="${STUDENT_WIDTH:-}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed1337.yaml}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set NORMAL_FRACTION explicitly: 0.0 or 0.25}"
RUN_ROOT="${RUN_ROOT:-$PACKAGE_ROOT/runs/four_arm_micro_${STUDENT_KIND}}"
PYTHON="${PYTHON:-python}"
DETERMINISM_MODE="${DETERMINISM_MODE:-best_effort}"
SMOKE_CRITIC_EPOCHS="${SMOKE_CRITIC_EPOCHS:-2}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
SMOKE_WARMUP="${SMOKE_WARMUP:-0}"
SMOKE_RAMP_EPOCHS="${SMOKE_RAMP_EPOCHS:-1}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

for f in "$MANIFEST" "$GATE0_CERTIFICATE" "$FULL_GATE0_CERTIFICATE" "$STUDENT_INIT" "$CONFIG"; do
  test -f "$f" || { echo "MISSING: $f" >&2; exit 2; }
done

if [ -z "$STUDENT_WIDTH" ]; then
  STUDENT_WIDTH="$($PYTHON - "$STUDENT_INIT" <<'PY'
import sys, torch
saved=torch.load(sys.argv[1], map_location="cpu", weights_only=False)
width=saved.get("student_width")
if width is None:
    raise SystemExit("student init missing student_width")
print(int(width))
PY
)"
fi

"$PYTHON" - "$NORMAL_FRACTION" <<'PY'
import sys
v=float(sys.argv[1])
if v not in (0.0, 0.25):
    raise SystemExit("smoke protocol requires NORMAL_FRACTION=0.0 or 0.25")
PY

"$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 \
  --config "$CONFIG" --manifest "$MANIFEST" --gate0-certificate "$GATE0_CERTIFICATE" \
  --full-gate0-certificate "$FULL_GATE0_CERTIFICATE" \
  --out "$RUN_ROOT/critic" --mode critic --critic-epochs "$SMOKE_CRITIC_EPOCHS" \
  --normal-fraction "$NORMAL_FRACTION" --normal-critic-weight 1.0 \
  --determinism-mode "$DETERMINISM_MODE"

CRITIC="$RUN_ROOT/critic/critic.pt"
METRICS="$RUN_ROOT/critic/critic_validation.json"
"$PYTHON" - "$METRICS" <<'PY'
import json,sys
from oasis_rc_v2.qualification import critic_gate_failures
failed=critic_gate_failures(json.load(open(sys.argv[1])))
if failed: raise SystemExit("critic quality gate failed: "+", ".join(failed))
PY

run() {
  local name="$1" mode="$2"; shift 2
  "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 \
    --config "$CONFIG" --manifest "$MANIFEST" --gate0-certificate "$GATE0_CERTIFICATE" \
    --full-gate0-certificate "$FULL_GATE0_CERTIFICATE" \
    --out "$RUN_ROOT/$name" --mode "$mode" --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" \
    --epochs "$SMOKE_EPOCHS" --warmup "$SMOKE_WARMUP" --ramp-epochs "$SMOKE_RAMP_EPOCHS" \
    --normal-fraction "$NORMAL_FRACTION" --determinism-mode "$DETERMINISM_MODE" \
    --student-init-checkpoint "$STUDENT_INIT" "$@"
}

run S0_control control
run S1_oasis_rc_v2 connected --lambda-oasis 0.001 --critic-checkpoint "$CRITIC"
run S2_aosk aosk --lambda-aosk 0.01
run S3_oasis_rc_v2_aosk aosk_connected --lambda-oasis 0.001 --lambda-aosk 0.01 --critic-checkpoint "$CRITIC"

echo "Four-arm validation micro-smoke complete: $RUN_ROOT"
echo "STUDENT_KIND=$STUDENT_KIND"
echo "STUDENT_WIDTH=$STUDENT_WIDTH"
echo "SMOKE_CRITIC_EPOCHS=$SMOKE_CRITIC_EPOCHS"
echo "SMOKE_EPOCHS=$SMOKE_EPOCHS"
echo "AOSK_VARIANT=oriented-consistency-v1"
echo "TEST_FIREWALL=CLOSED"
