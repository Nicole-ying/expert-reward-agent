#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/env001_reproduce_paper_v4.yaml"
PREFIX="rpv4"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20
SEEDS=(0 1 2 3 4)

echo "============================================================"
echo " Reproduce paper_v4 (subagent disabled)"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : ${SEEDS[*]}"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "START TIME      : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "##############################################################"
  echo " SEED $seed  —  $(date '+%H:%M:%S')"
  echo "##############################################################"
  python -m pipeline.run_iterative_experiment \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES"
  echo " SEED $seed DONE  —  $(date '+%H:%M:%S')"
done

echo ""
echo "============================================================"
echo " All seeds done.  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
