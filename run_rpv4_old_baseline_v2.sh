#!/usr/bin/env bash
set -euo pipefail
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export CUDA_VISIBLE_DEVICES=""
CONFIG="configs/env001_rpv4_old_baseline_v2.yaml"
PREFIX="rpv4_old_baseline_v2"
ROUNDS=10; TOTAL_TIMESTEPS=1000000; EVAL_EPISODES=20
echo "===== rpv4_old_baseline_v2: stateless, same v1 as paper_v4 ====="
echo "CONFIG=$CONFIG  SEEDS=0..4  ROUNDS=$ROUNDS  RESUME_FROM=2"
for seed in 0 1 2 3 4; do
  echo "SEED $seed — $(date '+%H:%M:%S')"
  python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed "$seed" --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES" --resume-from 2
  echo "SEED $seed DONE — $(date '+%H:%M:%S')"
done
echo "ALL DONE — $(date)"