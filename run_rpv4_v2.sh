#!/usr/bin/env bash
set -euo pipefail
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-C:/ProgramData/miniconda3/envs/eure/Lib/site-packages/certifi/cacert.pem}"
CONFIG="configs/env001_rpv4_v2.yaml"
PREFIX="rpv4_v2"
ROUNDS=10; TOTAL_TIMESTEPS=1000000; EVAL_EPISODES=20
echo "===== rpv4_v2: V2 narrative prompt ====="
echo "CONFIG=$CONFIG  SEEDS=0..4  ROUNDS=$ROUNDS"
for seed in 0 1 2 3 4; do
  echo "SEED $seed — $(date '+%H:%M:%S')"
  python -m pipeline.run_iterative_experiment --config "$CONFIG" --prefix "$PREFIX" --seed "$seed" --rounds "$ROUNDS" --total-timesteps "$TOTAL_TIMESTEPS" --eval-episodes "$EVAL_EPISODES"
  echo "SEED $seed DONE — $(date '+%H:%M:%S')"
done
echo "ALL DONE — $(date)"
