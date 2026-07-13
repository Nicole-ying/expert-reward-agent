#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env003_deepseek_rag.yaml"
PREFIX="exp_deres_cartpole_5seed"
LOG_DIR="runs/env_003/logs"

mkdir -p "$LOG_DIR"

python -u -m pipeline.run_multi_seed_experiment \
  --config "$CONFIG" \
  --prefix "$PREFIX" \
  --rounds 10 \
  --total-timesteps 100000 \
  --eval-episodes 20
