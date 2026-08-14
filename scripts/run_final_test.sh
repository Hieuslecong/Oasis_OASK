#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

LOCK="${1:?usage: scripts/run_final_test.sh /path/to/PROTOCOL_LOCK.json}"
MARKER="${LOCK}.test_opened"

# IMPORTANT: this first phase validates only lock/checkpoint/manifest files and then
# atomically creates OPENED. It intentionally does not open any dataset image/mask.
readarray -t FIELDS < <("$PYTHON" - "$LOCK" "$MARKER" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

lock_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
d = json.loads(lock_path.read_text())
required = (
    "selected_checkpoint",
    "selected_checkpoint_sha256",
    "manifest",
    "manifest_sha256",
    "dataset_content_sha256",
    "training_view_dataset_sha256",
    "full_gate0_certificate_sha256",
    "threshold",
    "hyperparameters_locked",
)
missing = [k for k in required if k not in d]
if missing:
    raise SystemExit("missing protocol-lock fields: " + ", ".join(missing))
if d["hyperparameters_locked"] is not True:
    raise SystemExit("hyperparameters_locked must be true")
threshold = float(d["threshold"])
if not 0.0 < threshold < 1.0:
    raise SystemExit("threshold must be in (0,1)")

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if sha(d["selected_checkpoint"]) != d["selected_checkpoint_sha256"]:
    raise SystemExit("checkpoint SHA mismatch")
if sha(d["manifest"]) != d["manifest_sha256"]:
    raise SystemExit("manifest SHA mismatch")

# Check that the locked threshold is exactly the validation-selected checkpoint threshold.
import torch
ck = torch.load(d["selected_checkpoint"], map_location="cpu", weights_only=False)
ck_threshold = float(ck.get("threshold_validation", -1.0))
if abs(ck_threshold - threshold) > 1e-12:
    raise SystemExit("lock threshold does not equal checkpoint threshold_validation")
if ck.get("dataset_content_sha256") is None:
    raise SystemExit("checkpoint missing dataset_content_sha256")
if ck.get("training_view_dataset_sha256") != d["training_view_dataset_sha256"]:
    raise SystemExit("lock training-view dataset SHA does not match checkpoint")
if ck.get("full_gate0_certificate_sha256") != d["full_gate0_certificate_sha256"]:
    raise SystemExit("lock full Gate 0 certificate SHA does not match checkpoint")

marker = {
    "state": "OPENED",
    "opened_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "protocol_lock": str(lock_path.resolve()),
    "protocol_lock_sha256": sha(lock_path),
    "checkpoint": d["selected_checkpoint"],
    "checkpoint_sha256": d["selected_checkpoint_sha256"],
    "manifest": d["manifest"],
    "manifest_sha256": d["manifest_sha256"],
    "dataset_content_sha256": d["dataset_content_sha256"],
    "training_view_dataset_sha256": d["training_view_dataset_sha256"],
    "full_gate0_certificate_sha256": d["full_gate0_certificate_sha256"],
    "threshold": threshold,
    "output": d.get("output", str(lock_path.with_name("final_test.json"))),
}
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
try:
    fd = os.open(marker_path, flags, 0o600)
except FileExistsError:
    raise SystemExit(f"REFUSE: canonical test was already opened: {marker_path}")
with os.fdopen(fd, "w") as f:
    json.dump(marker, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
print(d["selected_checkpoint"])
print(d["manifest"])
print(marker["output"])
print(f"{threshold:.17g}")
PY
)

CKPT="${FIELDS[0]}"
MANIFEST="${FIELDS[1]}"
OUT="${FIELDS[2]}"
THRESHOLD="${FIELDS[3]}"

# From this point on the canonical test is considered opened even if evaluation fails.
"$PYTHON" -m oasis_cycle_aosk.evaluate_rc \
  --checkpoint "$CKPT" \
  --manifest "$MANIFEST" \
  --split test \
  --threshold "$THRESHOLD" \
  --final-test-authorization "$MARKER" \
  --device cuda \
  --out "$OUT"

"$PYTHON" - "$MARKER" "$OUT" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

marker_path = Path(sys.argv[1])
out = Path(sys.argv[2])

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

m = json.loads(marker_path.read_text())
m["state"] = "DONE"
m["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
m["output_sha256"] = sha(out)
tmp = marker_path.with_suffix(marker_path.suffix + ".tmp")
tmp.write_text(json.dumps(m, indent=2))
os.replace(tmp, marker_path)
PY

echo "FINAL_TEST_DONE $OUT"
