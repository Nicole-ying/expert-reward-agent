#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env001_ablation_unconstrained_v4.yaml"
PREFIX="ablation_unconstrained_v4"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20

echo "============================================================"
echo " Ablation A: Unconstrained Reflection (No L1/L2/L3)"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : 5"
echo "ROUNDS          : $ROUNDS"
echo "Memory          : ON (v4 setting)"
echo "RAG             : OFF in reflection (v4 setting)"
echo "Feedback        : structured (component-level)"
echo "Reflection      : UNCONSTRAINED"
echo ""

for seed in 0 1 2 3 4; do
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
