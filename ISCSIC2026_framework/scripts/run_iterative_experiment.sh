#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/env001_deepseek_rag.yaml}
PREFIX=${2:-}
MOCK_FLAG=${3:-}

if [[ -n "$PREFIX" && "$PREFIX" != "--mock" ]]; then
  PREFIX_ARGS=(--prefix "$PREFIX")
else
  PREFIX_ARGS=()
fi

if [[ "$PREFIX" == "--mock" || "$MOCK_FLAG" == "--mock" ]]; then
  MOCK_ARGS=(--mock)
else
  MOCK_ARGS=()
fi

python -m pipeline.run_iterative_experiment \
  --config "$CONFIG" \
  "${PREFIX_ARGS[@]}" \
  "${MOCK_ARGS[@]}"
