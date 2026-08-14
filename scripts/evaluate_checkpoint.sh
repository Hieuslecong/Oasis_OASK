#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:?Usage: scripts/evaluate_checkpoint.sh checkpoint.pt manifest.jsonl split threshold output.json}"
MANIFEST="${2:?Missing manifest}"
SPLIT="${3:?Missing split}"
THRESHOLD="${4:?Missing validation-selected threshold}"
OUTPUT="${5:?Missing output JSON}"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$PACKAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "$SPLIT" = "test" ]; then
  echo "REFUSE: canonical test is only accessible through scripts/run_final_test.sh with PROTOCOL_LOCK.json" >&2
  exit 3
fi

# Do not pass --size here. evaluate_rc reads the training resolution from the
# checkpoint and rejects accidental resolution changes unless an explicit
# ablation flag is supplied by the caller.
"$PYTHON" -m oasis_cycle_aosk.evaluate_rc \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --split "$SPLIT" \
  --threshold "$THRESHOLD" \
  --out "$OUTPUT"
