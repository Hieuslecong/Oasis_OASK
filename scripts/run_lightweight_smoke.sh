#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:?Usage: scripts/run_lightweight_smoke.sh /absolute/path/to/manifest.jsonl}"

for STUDENT_KIND in mobilenetv3 dsunet fastscnn bisenet; do
  "$PACKAGE_ROOT/scripts/run_smoke.sh" "$MANIFEST" "$STUDENT_KIND"
done
