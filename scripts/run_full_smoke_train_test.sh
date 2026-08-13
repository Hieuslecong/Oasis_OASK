#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACKAGE_ROOT"

# Required real-data inputs.
CANONICAL_MANIFEST="${CANONICAL_MANIFEST:?set CANONICAL_MANIFEST=/absolute/path/to/canonical_manifest.jsonl}"
NORMAL_ROOT="${NORMAL_ROOT:?set NORMAL_ROOT=/absolute/path/to/true_normal_rgb}"

# One-shot smoke configuration.
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
SEED="${SEED:-1337}"
CONFIG="${CONFIG:-$PACKAGE_ROOT/configs/canonical_gpu_256_seed${SEED}.yaml}"
EXP_ROOT="${EXP_ROOT:-$PACKAGE_ROOT/runs/full_smoke_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-$EXP_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$EXP_ROOT/four_arm_smoke}"
NORMAL_FRACTION="${NORMAL_FRACTION:-0.25}"
STUDENT_KIND="${STUDENT_KIND:-multiscale}"
STUDENT_WIDTH="${STUDENT_WIDTH:-16}"
DETERMINISM_MODE="${DETERMINISM_MODE:-best_effort}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"
PREPARE_DATA="${PREPARE_DATA:-1}"
SMOKE_TEST_VAL_SAMPLES="${SMOKE_TEST_VAL_SAMPLES:-8}"
SMOKE_TEST_NORMAL_SAMPLES="${SMOKE_TEST_NORMAL_SAMPLES:-4}"

export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DATA_ROOT CANONICAL_MANIFEST NORMAL_ROOT PYTHON

mkdir -p "$EXP_ROOT" "$RUN_ROOT"

fail() {
  echo "FULL_SMOKE_FAIL: $*" >&2
  exit 1
}

for f in "$CANONICAL_MANIFEST" "$CONFIG"; do
  test -f "$f" || fail "missing file: $f"
done
test -d "$NORMAL_ROOT" || fail "missing normal RGB directory: $NORMAL_ROOT"
test -x "$PYTHON" || fail "python is not executable: $PYTHON"

echo "============================================================"
echo "OASIS-RC v2 FULL TRAIN + SMOKE-TEST"
echo "============================================================"
echo "CANONICAL_MANIFEST=$CANONICAL_MANIFEST"
echo "NORMAL_ROOT=$NORMAL_ROOT"
echo "EXP_ROOT=$EXP_ROOT"
echo "DATA_ROOT=$DATA_ROOT"
echo "SEED=$SEED"
echo "NORMAL_FRACTION=$NORMAL_FRACTION"
echo "STUDENT_KIND=$STUDENT_KIND"
echo "DETERMINISM_MODE=$DETERMINISM_MODE"
echo "CANONICAL_TEST_POLICY=CLOSED"

# ---------------------------------------------------------------------------
# 0. Environment / code smoke.
# ---------------------------------------------------------------------------
echo "== SMOKE 0/7 environment =="
"$PYTHON" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import torch, yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("config_device:", cfg.get("device"))
print("cuda_available:", torch.cuda.is_available())
if str(cfg.get("device", "cpu")).startswith("cuda"):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA config requested but torch.cuda.is_available() is false")
    print("gpu:", torch.cuda.get_device_name(0))
    print("torch_cuda:", torch.version.cuda)
    print("cudnn:", torch.backends.cudnn.version())
PY

"$PYTHON" -m compileall -q src scripts tests
if [ "$RUN_UNIT_TESTS" = "1" ]; then
  "$PYTHON" -m pytest -q
fi

# ---------------------------------------------------------------------------
# 1. Real-data preparation / Gate 0.
# ---------------------------------------------------------------------------
echo "== SMOKE 1/7 real-data preparation + Gate 0 =="
if [ "$PREPARE_DATA" = "1" ]; then
  bash "$PACKAGE_ROOT/scripts/prepare_real_data.sh" \
    2>&1 | tee "$EXP_ROOT/prepare_real_data.log"
fi

N0_MODE="$($PYTHON - "$NORMAL_FRACTION" <<'PY'
import sys
print("1" if abs(float(sys.argv[1])) < 1e-12 else "0")
PY
)"

if [ "$N0_MODE" = "1" ]; then
  TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_n0.jsonl"
  GATE0_CERTIFICATE="$DATA_ROOT/gate0_training_n0.json"
else
  TRAIN_MANIFEST="$DATA_ROOT/manifest_trainval_with_normal.jsonl"
  GATE0_CERTIFICATE="$DATA_ROOT/gate0_training.json"
fi

for f in "$TRAIN_MANIFEST" "$GATE0_CERTIFICATE"; do
  test -f "$f" || fail "prepared training artifact is missing: $f"
done

echo "TRAIN_MANIFEST=$TRAIN_MANIFEST"
echo "GATE0_CERTIFICATE=$GATE0_CERTIFICATE"

# ---------------------------------------------------------------------------
# 2. Canonical shared student initialization.
# ---------------------------------------------------------------------------
echo "== SMOKE 2/7 shared student init =="
STUDENT_INIT="$EXP_ROOT/init/student_init_seed${SEED}.pt"
mkdir -p "$(dirname "$STUDENT_INIT")"
"$PYTHON" "$PACKAGE_ROOT/scripts/create_student_init.py" \
  --seed "$SEED" \
  --student-kind "$STUDENT_KIND" \
  --student-width "$STUDENT_WIDTH" \
  --out "$STUDENT_INIT"

# ---------------------------------------------------------------------------
# 3. Critic + four-arm training smoke (2 epochs each in run_smoke.sh).
# ---------------------------------------------------------------------------
echo "== SMOKE 3/7 critic + S0/S1/S2/S3 training =="
CONFIG="$CONFIG" \
NORMAL_FRACTION="$NORMAL_FRACTION" \
RUN_ROOT="$RUN_ROOT" \
PYTHON="$PYTHON" \
DETERMINISM_MODE="$DETERMINISM_MODE" \
bash "$PACKAGE_ROOT/scripts/run_smoke.sh" \
  "$TRAIN_MANIFEST" \
  "$GATE0_CERTIFICATE" \
  "$STUDENT_INIT" \
  "$STUDENT_KIND" \
  2>&1 | tee "$EXP_ROOT/four_arm_training.log"

# ---------------------------------------------------------------------------
# 4. Build a non-canonical smoke-test split.
#    IMPORTANT: this copies a few VAL / normal_train rows into a new manifest
#    named smoke_test. It never opens or copies split=test from the benchmark.
# ---------------------------------------------------------------------------
echo "== SMOKE 4/7 build isolated smoke_test manifest =="
SMOKE_MANIFEST="$EXP_ROOT/smoke_test_manifest.jsonl"
"$PYTHON" - "$TRAIN_MANIFEST" "$SMOKE_MANIFEST" \
  "$SMOKE_TEST_VAL_SAMPLES" "$SMOKE_TEST_NORMAL_SAMPLES" <<'PY'
import json, sys
from pathlib import Path

src, dst = map(Path, sys.argv[1:3])
val_n, normal_n = map(int, sys.argv[3:5])
rows = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]
if any(row.get("split") == "test" for row in rows):
    raise SystemExit("training-view manifest unexpectedly contains canonical test rows")

val = sorted((r for r in rows if r.get("split") == "val"), key=lambda r: str(r.get("image", "")))
normal = sorted((r for r in rows if r.get("split") == "normal_train"), key=lambda r: str(r.get("image", "")))
selected = val[:val_n] + normal[:normal_n]
if not selected:
    raise SystemExit("cannot construct smoke_test: no val/normal_train rows")

out = []
for row in selected:
    copy = dict(row)
    copy["smoke_source_split"] = row.get("split")
    copy["split"] = "smoke_test"
    out.append(copy)

dst.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out))
print(json.dumps({
    "smoke_test_manifest": str(dst),
    "rows": len(out),
    "from_val": min(len(val), val_n),
    "from_normal_train": min(len(normal), normal_n),
    "canonical_test_rows_opened": 0,
}, indent=2))
PY

# ---------------------------------------------------------------------------
# 5. End-to-end inference/evaluator smoke on all four deployment checkpoints.
# ---------------------------------------------------------------------------
echo "== SMOKE 5/7 evaluate S0/S1/S2/S3 on smoke_test =="
EVAL_DEVICE="${EVAL_DEVICE:-$($PYTHON - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
print(yaml.safe_load(Path(sys.argv[1]).read_text()).get("device", "cpu"))
PY
)}"

SMOKE_RESULT_ROOT="$EXP_ROOT/smoke_test_results"
mkdir -p "$SMOKE_RESULT_ROOT"

ARMS=(
  "S0_control"
  "S1_oasis"
  "S2_aosk_oriented"
  "S3_oasis_aosk_oriented"
)

for arm in "${ARMS[@]}"; do
  checkpoint="$RUN_ROOT/$arm/student_only.pt"
  test -f "$checkpoint" || fail "missing deployment checkpoint: $checkpoint"
  "$PYTHON" -m oasis_cycle_aosk.evaluate_rc \
    --checkpoint "$checkpoint" \
    --manifest "$SMOKE_MANIFEST" \
    --split smoke_test \
    --device "$EVAL_DEVICE" \
    --out "$SMOKE_RESULT_ROOT/$arm.json"
done

# ---------------------------------------------------------------------------
# 6. Assert all smoke outputs and print compact summary.
# ---------------------------------------------------------------------------
echo "== SMOKE 6/7 result assertions =="
"$PYTHON" - "$SMOKE_RESULT_ROOT" <<'PY'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
arms = [
    "S0_control",
    "S1_oasis",
    "S2_aosk_oriented",
    "S3_oasis_aosk_oriented",
]
print(f"{'ARM':30s} {'DICE/F1':>10s} {'IOU':>10s} {'THRESH':>10s}")
for arm in arms:
    path = root / f"{arm}.json"
    if not path.exists():
        raise SystemExit(f"missing smoke result: {path}")
    result = json.loads(path.read_text())
    if result.get("split") != "smoke_test":
        raise SystemExit(f"unexpected split for {arm}: {result.get('split')}")
    if result.get("final_test_authorized") is not False:
        raise SystemExit(f"smoke evaluation unexpectedly used final-test authorization: {arm}")
    for key in ("dice_f1", "iou", "threshold"):
        value = float(result[key])
        if not math.isfinite(value):
            raise SystemExit(f"non-finite {key} for {arm}: {value}")
    print(f"{arm:30s} {result['dice_f1']:10.6f} {result['iou']:10.6f} {result['threshold']:10.4f}")
PY

# ---------------------------------------------------------------------------
# 7. Final contract check.
# ---------------------------------------------------------------------------
echo "== SMOKE 7/7 complete =="
echo "FULL_SMOKE_PASS"
echo "TRAINING_ARMS=S0,S1,S2,S3"
echo "SMOKE_TEST_SPLIT=smoke_test"
echo "CANONICAL_TEST_OPENED=NO"
echo "TEST_FIREWALL=CLOSED"
echo "EXP_ROOT=$EXP_ROOT"
