#!/usr/bin/env bash
set -euo pipefail

echo "scripts/run_three_round_experiment.sh is deprecated."
echo "Use scripts/run_iterative_experiment.sh with iteration.total_rounds in config."

exec bash scripts/run_iterative_experiment.sh "$@"
