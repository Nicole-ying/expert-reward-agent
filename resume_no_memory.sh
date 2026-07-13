#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env001_deepseek_rag.yaml"
PREFIX="ablation_no_memory_v1"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20

echo "============================================================"
echo " Resume: ablation_no_memory_v1 (no memory)"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo ""

# Seed 0: resume from iter 4
echo ""
echo "##############################################################"
echo " SEED 0 / 5 -- RESUME from iter 4"
echo "##############################################################"
python -m pipeline.run_iterative_experiment \
  --config "$CONFIG" \
  --prefix "$PREFIX" \
  --seed 0 \
  --resume-from 4 \
  --rounds "$ROUNDS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES"

# Seeds 1-4: fresh start
for seed in 1 2 3 4; do
  echo ""
  echo "##############################################################"
  echo " SEED $seed / 5"
  echo "##############################################################"
  python -m pipeline.run_iterative_experiment \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES"
done

echo ""
echo "============================================================"
echo " All seeds done."
echo "============================================================"
