#!/usr/bin/env bash
# One-command wrapper for the interactive autoresearch experiment tree.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-docs/autoresearch_dashboard/index.html}"
python3 scripts/generate_experiment_tree_web.py --out "$OUT"

printf 'Open: %s/%s\n' "$(pwd)" "$OUT"
printf 'For editable user summaries, run: python3 scripts/serve_dashboard.py\n'
