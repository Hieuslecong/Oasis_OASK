#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST}"
NORMAL_ROOT="${NORMAL_ROOT:-}"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
SEED="${SEED:-1337}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml}"
EXP_ROOT="${EXP_ROOT:-$PACKAGE_ROOT/runs/full_smoke_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$EXP_ROOT/four_arm_smoke}"
NORMAL_FRACTION="${NORMAL_FRACTION:?set NORMAL_FRACTION explicitly: 0.0 or 0.25}"
STUDENT_KIND="${STUDENT_KIND:-multiscale}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
DETERMINISM_MODE="${DETERMINISM_MODE:-best_effort}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
SMOKE_CRITIC_EPOCHS="${SMOKE_CRITIC_EPOCHS:-10}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
SMOKE_WARMUP="${SMOKE_WARMUP:-0}"
SMOKE_RAMP_EPOCHS="${SMOKE_RAMP_EPOCHS:-1}"
SMOKE_TEST_VAL_SAMPLES="${SMOKE_TEST_VAL_SAMPLES:-8}"
SMOKE_TEST_NORMAL_SAMPLES="${SMOKE_TEST_NORMAL_SAMPLES:-4}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DATA_ROOT CANONICAL_MANIFEST NORMAL_ROOT PYTHON
mkdir -p "$EXP_ROOT" "$RUN_ROOT"

fail() { echo "FULL_SMOKE_FAIL: $*" >&2; exit 1; }
for f in "$CANONICAL_MANIFEST" "$CONFIG"; do test -f "$f" || fail "missing $f"; done
test -x "$PYTHON" || fail "python is not executable: $PYTHON"

PROTOCOL="$($PYTHON - "$NORMAL_FRACTION" <<'PY'
import sys
v=float(sys.argv[1])
if v == 0.0: print("N0")
elif v == 0.25: print("N25")
else: raise SystemExit("full smoke requires NORMAL_FRACTION=0.0 or 0.25")
PY
)"
if [ "$PROTOCOL" = "N25" ]; then
  test -n "$NORMAL_ROOT" || fail "N25 requires NORMAL_ROOT"
  test -d "$NORMAL_ROOT" || fail "missing normal RGB directory: $NORMAL_ROOT"
fi

echo "== SMOKE 0/7 environment + tests =="
"$PYTHON" - "$CONFIG" <<'PY'
import sys, torch, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
print("python",sys.version.split()[0]); print("torch",torch.__version__)
print("device",cfg.get("device")); print("cuda_available",torch.cuda.is_available())
if str(cfg.get("device","cpu")).startswith("cuda"):
    if not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    print("gpu",torch.cuda.get_device_name(0)); print("cuda",torch.version.cuda)
PY
"$PYTHON" -m compileall -q src scripts tests
if [ "$RUN_UNIT_TESTS" = "1" ]; then "$PYTHON" -m pytest -q; fi

echo "== SMOKE 1/7 data preparation + Gate 0 ($PROTOCOL) =="
if [ "$PREPARE_DATA" = "1" ]; then
  if [ "$PROTOCOL" = "N0" ]; then
    bash "$PACKAGE_ROOT/scripts/prepare_n0_data.sh" 2>&1 | tee "$EXP_ROOT/prepare_data.log"
  else
    bash "$PACKAGE_ROOT/scripts/prepare_real_data.sh" 2>&1 | tee "$EXP_ROOT/prepare_data.log"
  fi
fi
if [ "$PROTOCOL" = "N0" ]; then
  TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_n0.jsonl"
  GATE0_CERTIFICATE="$DATA_ROOT/gate0_training_n0.json"
  FULL_GATE0_CERTIFICATE="$DATA_ROOT/gate0_full_n0.json"
else
  TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_with_normal.jsonl"
  GATE0_CERTIFICATE="$DATA_ROOT/gate0_training.json"
  FULL_GATE0_CERTIFICATE="$DATA_ROOT/gate0_full.json"
fi
for f in "$TRAIN_MANIFEST" "$GATE0_CERTIFICATE" "$FULL_GATE0_CERTIFICATE"; do test -f "$f" || fail "missing $f"; done

echo "== SMOKE 2/7 shared student init =="
STUDENT_INIT="$EXP_ROOT/init/student_init_seed${SEED}.pt"
mkdir -p "$(dirname "$STUDENT_INIT")"
"$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" \
  --seed "$SEED" --student-kind "$STUDENT_KIND" --student-width "$STUDENT_WIDTH" --out "$STUDENT_INIT"

echo "== SMOKE 3/7 critic + S0/S1/S2/S3 =="
CONFIG="$CONFIG" NORMAL_FRACTION="$NORMAL_FRACTION" RUN_ROOT="$RUN_ROOT" PYTHON="$PYTHON" \
DETERMINISM_MODE="$DETERMINISM_MODE" SMOKE_CRITIC_EPOCHS="$SMOKE_CRITIC_EPOCHS" \
SMOKE_EPOCHS="$SMOKE_EPOCHS" SMOKE_WARMUP="$SMOKE_WARMUP" \
SMOKE_RAMP_EPOCHS="$SMOKE_RAMP_EPOCHS" \
bash "$PACKAGE_ROOT/scripts/run_smoke.sh" "$TRAIN_MANIFEST" "$GATE0_CERTIFICATE" "$STUDENT_INIT" "$STUDENT_KIND" "$FULL_GATE0_CERTIFICATE" \
  2>&1 | tee "$EXP_ROOT/four_arm_training.log"

echo "== SMOKE 4/7 build non-canonical smoke_test =="
SMOKE_MANIFEST="$EXP_ROOT/smoke_test_manifest.jsonl"
"$PYTHON" - "$TRAIN_MANIFEST" "$SMOKE_MANIFEST" "$SMOKE_TEST_VAL_SAMPLES" "$SMOKE_TEST_NORMAL_SAMPLES" <<'PY'
import json,sys
from pathlib import Path
src,dst=map(Path,sys.argv[1:3]); val_n,normal_n=map(int,sys.argv[3:5])
rows=[json.loads(x) for x in src.read_text().splitlines() if x.strip()]
if any(r.get("split")=="test" for r in rows): raise SystemExit("canonical test row found in training view")
val=sorted((r for r in rows if r.get("split")=="val"),key=lambda r:str(r.get("image","")))
normal=sorted((r for r in rows if r.get("split")=="normal_train"),key=lambda r:str(r.get("image","")))
selected=val[:val_n]+normal[:normal_n]
if not selected: raise SystemExit("cannot build smoke_test")
out=[]
for r in selected:
    c=dict(r); c["smoke_source_split"]=r.get("split"); c["split"]="smoke_test"; out.append(c)
dst.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out))
print({"rows":len(out),"canonical_test_rows_opened":0})
PY

echo "== SMOKE 5/7 evaluate deployment checkpoints =="
EVAL_DEVICE="${EVAL_DEVICE:-$($PYTHON - "$CONFIG" <<'PY'
import sys,yaml
from pathlib import Path
print(yaml.safe_load(Path(sys.argv[1]).read_text()).get("device","cpu"))
PY
)}"
SMOKE_RESULT_ROOT="$EXP_ROOT/smoke_test_results"; mkdir -p "$SMOKE_RESULT_ROOT"
ARMS=(S0_control S1_oasis_rc_v2 S2_aosk S3_oasis_rc_v2_aosk)
for arm in "${ARMS[@]}"; do
  checkpoint="$RUN_ROOT/$arm/student_only.pt"
  test -f "$checkpoint" || fail "missing deployment checkpoint: $checkpoint"
  "$PYTHON" -m oasis_cycle_aosk.evaluate_rc --checkpoint "$checkpoint" --manifest "$SMOKE_MANIFEST" \
    --split smoke_test --device "$EVAL_DEVICE" --out "$SMOKE_RESULT_ROOT/$arm.json"
done

echo "== SMOKE 6/7 assertions =="
"$PYTHON" - "$SMOKE_RESULT_ROOT" <<'PY'
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1]); arms=["S0_control","S1_oasis_rc_v2","S2_aosk","S3_oasis_rc_v2_aosk"]
print(f"{'ARM':30s} {'DICE/F1':>10s} {'IOU':>10s} {'THRESH':>10s}")
for arm in arms:
    r=json.loads((root/f"{arm}.json").read_text())
    if r.get("split")!="smoke_test" or r.get("final_test_authorized") is not False: raise SystemExit(arm)
    for k in ("dice_f1","iou","threshold"):
        if not math.isfinite(float(r[k])): raise SystemExit(f"non-finite {arm} {k}")
    print(f"{arm:30s} {r['dice_f1']:10.6f} {r['iou']:10.6f} {r['threshold']:10.4f}")
PY

echo "== SMOKE 7/7 complete =="
echo "FULL_SMOKE_PASS"
echo "TRAINING_VARIANT=$PROTOCOL"
echo "AOSK_VARIANT=oriented-consistency-v1"
echo "SMOKE_TEST_SPLIT=smoke_test"
echo "CANONICAL_TEST_OPENED=NO"
echo "TEST_FIREWALL=CLOSED"
echo "EXP_ROOT=$EXP_ROOT"
