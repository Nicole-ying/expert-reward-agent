#!/usr/bin/env bash
set -euo pipefail
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
CONFIG="configs/env001_rpv4_old_v3.yaml"
PREFIX="rpv4_old_v3"
ROUNDS=10; TOTAL_TIMESTEPS=1000000; EVAL_EPISODES=20
echo "===== rpv4_old_v3: resume/restart ====="
echo "CONFIG=$CONFIG  ROUNDS=$ROUNDS"

# seed_0: completed iter_04, resume from 5
echo "SEED 0 (resume from 5) — $(date '+%H:%M:%S')"
python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed 0 --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES" --resume-from 5
echo "SEED 0 DONE — $(date '+%H:%M:%S')"

# seed_1: completed iter_05, resume from 6
echo "SEED 1 (resume from 6) — $(date '+%H:%M:%S')"
python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed 1 --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES" --resume-from 6
echo "SEED 1 DONE — $(date '+%H:%M:%S')"

# seeds 2-4: fresh start
for seed in 2 3 4; do
  echo "SEED $seed (fresh) — $(date '+%H:%M:%S')"
  python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed "$seed" --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES"
  echo "SEED $seed DONE — $(date '+%H:%M:%S')"
done
echo "ALL DONE — $(date)"