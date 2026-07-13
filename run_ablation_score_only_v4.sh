#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env001_ablation_score_only_v4.yaml"
PREFIX="ablation_score_only_v4"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20

echo "============================================================"
echo " Ablation: Score-Only Feedback (No Component Evidence Table)"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "SEEDS           : 5"
echo "ROUNDS          : $ROUNDS"
echo "Memory          : ON"
echo "Reflection      : structured (L1/L2/L3)"
echo "Feedback        : SCORE-ONLY (no magnitude_share/active_rate)"
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
