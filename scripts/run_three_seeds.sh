#!/usr/bin/env bash
set -euo pipefail

echo "DEPRECATED: scripts/run_three_seeds.sh was a smoke runner and is not a canonical 3-seed experiment." >&2
echo "Use scripts/run_all_seeds.sh with explicit CANONICAL_MANIFEST, BASE_EXP_ROOT, and NORMAL_FRACTION=0.0 (N0) or 0.25 (N25)." >&2
echo "N25 additionally requires NORMAL_ROOT; N0 does not." >&2
exit 64
