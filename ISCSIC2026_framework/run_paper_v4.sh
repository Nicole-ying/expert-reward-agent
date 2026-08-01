#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env001_paper_v4.yaml"
PREFIX="paper_v4"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20

echo "============================================================"
echo " Paper v4 Main Experiment"
echo "  Fix: reflection prompt excludes sections 9-12"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : 5"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
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
