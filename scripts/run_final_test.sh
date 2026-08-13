#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PYTHON="${PYTHON:-/hdd1/hieulc/Oasis_AOSK/.venv-oasis-rc-v2-gpu/bin/python}"; export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
LOCK="${1:?usage: scripts/run_final_test.sh /path/to/PROTOCOL_LOCK.json}"; MARKER="${LOCK}.test_opened"
if [ -e "$MARKER" ] && [ "${FORCE_FINAL_TEST:-0}" != "1" ]; then echo "REFUSE: final test already opened according to $MARKER" >&2; exit 3; fi
readarray -t FIELDS < <("$PYTHON" - "$LOCK" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text());required=("selected_checkpoint","selected_checkpoint_sha256","manifest","manifest_sha256","hyperparameters_locked");missing=[k for k in required if k not in d]
if missing:raise SystemExit("missing protocol-lock fields: "+", ".join(missing))
if d["hyperparameters_locked"] is not True:raise SystemExit("hyperparameters_locked must be true")
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
if sha(d["selected_checkpoint"])!=d["selected_checkpoint_sha256"]:raise SystemExit("checkpoint SHA mismatch")
if sha(d["manifest"])!=d["manifest_sha256"]:raise SystemExit("manifest SHA mismatch")
print(d["selected_checkpoint"]);print(d["manifest"]);print(d.get("output",str(p.with_name("final_test.json"))))
PY
)
CKPT="${FIELDS[0]}"; MANIFEST="${FIELDS[1]}"; OUT="${FIELDS[2]}"
"$PYTHON" -m oasis_cycle_aosk.evaluate_rc --checkpoint "$CKPT" --manifest "$MANIFEST" --split test --device cuda --out "$OUT"
{ echo "opened_utc=$(date -u +%FT%TZ)"; echo "checkpoint=$CKPT"; echo "manifest=$MANIFEST"; echo "output=$OUT"; } > "$MARKER"
echo "FINAL_TEST_DONE $OUT"
