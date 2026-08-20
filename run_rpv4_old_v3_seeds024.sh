#!/usr/bin/env bash
set -euo pipefail
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
CONFIG="configs/env001_rpv4_old_v3.yaml"
PREFIX="rpv4_old_v3"
ROUNDS=10; TOTAL_TIMESTEPS=1000000; EVAL_EPISODES=20
echo "===== rpv4_old_v3: seeds 0,2,3,4 ====="

# seed_0: resume from iter_05
echo "SEED 0 (resume from 5) — $(date '+%H:%M:%S')"
python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed 0 --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES" --resume-from 5
echo "SEED 0 DONE — $(date '+%H:%M:%S')"

# seeds 2-4: fresh start
for seed in 2 3 4; do
  echo "SEED $seed (fresh) — $(date '+%H:%M:%S')"
  python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed "$seed" --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES"
  echo "SEED $seed DONE — $(date '+%H:%M:%S')"
done
echo "ALL DONE — $(date)"
