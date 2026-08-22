#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
CONFIG="configs/env001_rpv4_old_claude.yaml"
PREFIX="rpv4_old_claude"
ROUNDS=10; TOTAL_TIMESTEPS=1000000; EVAL_EPISODES=20
echo "===== Claude Sonnet 5: paper_v4 prompt + behavior analyst ====="
echo "CONFIG=$CONFIG  SEEDS=0..4  ROUNDS=$ROUNDS  CPU"
for seed in 0 1 2 3 4; do
  echo "SEED $seed — $(date '+%H:%M:%S')"
  python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed "$seed" --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES"
  echo "SEED $seed DONE — $(date '+%H:%M:%S')"
done
echo "ALL DONE — $(date)"
