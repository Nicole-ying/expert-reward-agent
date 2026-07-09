#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env001_ablation_unconstrained_reflection.yaml"
PREFIX="ablation_unconstrained_v1"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20

echo "============================================================"
echo " Ablation: Unconstrained Reflection (NO L1/L2/L3)"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo "Memory          : OFF (inherited from base config)"
echo "RAG             : ON"
echo "Reflection      : UNCONSTRAINED"
echo ""

for seed in 0 1 2 3 4; do
  echo ""
  echo "##############################################################"
  echo " SEED $seed / 5"
  echo "##############################################################"
  if [ "$seed" -eq 0 ]; then
    python -m pipeline.run_iterative_experiment \
      --config "$CONFIG" \
      --prefix "$PREFIX" \
      --seed "$seed" \
      --resume-from 6 \
      --rounds "$ROUNDS" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --eval-episodes "$EVAL_EPISODES"
  else
    python -m pipeline.run_iterative_experiment \
      --config "$CONFIG" \
      --prefix "$PREFIX" \
      --seed "$seed" \
      --rounds "$ROUNDS" \
      --total-timesteps "$TOTAL_TIMESTEPS" \
      --eval-episodes "$EVAL_EPISODES"
  fi
done

echo ""
echo "============================================================"
echo " All seeds done."
echo "============================================================"
