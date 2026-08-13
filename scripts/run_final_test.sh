#!/usr/bin/env bash
# Single final-test entrypoint. Requires an explicit protocol lock and prevents
# accidental repeated test opening unless FORCE_FINAL_TEST=1 is set.
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

LOCK="${1:?usage: scripts/run_final_test.sh /path/to/PROTOCOL_LOCK.json}"
MARKER="${LOCK}.test_opened"

if [ -e "$MARKER" ] && [ "${FORCE_FINAL_TEST:-0}" != "1" ]; then
  echo "REFUSE: final test already opened according to $MARKER" >&2
  exit 3
fi

readarray -t FIELDS < <("$PYTHON" - "$LOCK" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
d = json.loads(p.read_text())
required = ("selected_checkpoint", "manifest", "hyperparameters_locked")
missing = [k for k in required if k not in d]
if missing:
    raise SystemExit("missing protocol-lock fields: " + ", ".join(missing))
if d["hyperparameters_locked"] is not True:
    raise SystemExit("hyperparameters_locked must be true")
print(d["selected_checkpoint"])
print(d["manifest"])
print(d.get("output", str(p.with_name("final_test.json"))))
PY
)

CKPT="${FIELDS[0]}"
MANIFEST="${FIELDS[1]}"
OUT="${FIELDS[2]}"
test -f "$CKPT"
test -f "$MANIFEST"

"$PYTHON" -m oasis_cycle_aosk.evaluate_rc \
  --checkpoint "$CKPT" --manifest "$MANIFEST" --split test \
  --device cuda --out "$OUT"

{
  echo "opened_utc=$(date -u +%FT%TZ)"
  echo "checkpoint=$CKPT"
  echo "manifest=$MANIFEST"
  echo "output=$OUT"
} > "$MARKER"

echo "FINAL_TEST_DONE $OUT"
