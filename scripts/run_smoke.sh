#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_smoke.sh trainval_manifest.jsonl gate0_training.json student_init.pt [student_kind]}"
GATE0_CERTIFICATE="${2:?Missing Gate 0 training-view certificate}"
STUDENT_INIT="${3:?Missing canonical student init checkpoint}"
STUDENT_KIND="${4:-multiscale}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed1337.yaml}"
NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}"
RUN_ROOT="${RUN_ROOT:-$PACKAGE_ROOT/runs/four_arm_micro_${STUDENT_KIND}}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
for f in "$MANIFEST" "$GATE0_CERTIFICATE" "$STUDENT_INIT" "$CONFIG"; do test -f "$f"; done
"$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 --config "$CONFIG" --manifest "$MANIFEST" --gate0-certificate "$GATE0_CERTIFICATE" --out "$RUN_ROOT/critic" --mode critic --critic-epochs 2 --normal-fraction "$NORMAL_FRACTION" --normal-critic-weight 1.0 --deterministic
CRITIC="$RUN_ROOT/critic/critic.pt"; METRICS="$RUN_ROOT/critic/critic_validation.json"
"$PYTHON" - "$METRICS" <<'PY'
import json,sys
from oasis_rc_v2.qualification import critic_gate_failures
failed=critic_gate_failures(json.load(open(sys.argv[1])))
if failed: raise SystemExit("critic quality gate failed: "+", ".join(failed))
PY
run(){ local name="$1" mode="$2"; shift 2; "$PYTHON" -m oasis_cycle_aosk.train_oasis_rc_v2 --config "$CONFIG" --manifest "$MANIFEST" --gate0-certificate "$GATE0_CERTIFICATE" --out "$RUN_ROOT/$name" --mode "$mode" --student-kind "$STUDENT_KIND" --epochs 2 --warmup 0 --ramp-epochs 1 --normal-fraction "$NORMAL_FRACTION" --deterministic --student-init-checkpoint "$STUDENT_INIT" "$@"; }
run S0_control control
run S2_aosk aosk --lambda-aosk 0.01
run S1_oasis connected --lambda-oasis 0.001 --critic-checkpoint "$CRITIC"
run S3_oasis_aosk aosk_connected --lambda-oasis 0.001 --lambda-aosk 0.01 --critic-checkpoint "$CRITIC"
echo "Four-arm canonical validation micro-smoke complete: $RUN_ROOT"; echo "TEST_FIREWALL=CLOSED"
