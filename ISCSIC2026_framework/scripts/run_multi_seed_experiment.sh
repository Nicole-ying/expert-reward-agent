#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/env001_deepseek_rag.yaml}
PREFIX=${2:-exp_iter}
NUM_SEEDS=${3:-10}
ROUNDS=${4:-10}
TOTAL_TIMESTEPS=${5:-1000000}
EVAL_EPISODES=${6:-10}
MOCK_FLAG=${7:-}

MOCK_ARGS=""
if [[ "$MOCK_FLAG" == "--mock" ]]; then
  MOCK_ARGS="--mock"
fi

echo "============================================================"
echo " Multi-seed iterative experiment"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "NUM_SEEDS       : $NUM_SEEDS"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo "MOCK            : ${MOCK_FLAG:-false}"
echo ""

for ((seed=0; seed<NUM_SEEDS; seed++)); do
  echo ""
  echo "##############################################################"
  echo " SEED $seed / $NUM_SEEDS"
  echo "##############################################################"
  python -m pipeline.run_iterative_experiment \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --rounds "$ROUNDS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --seed "$seed" \
    $MOCK_ARGS
done

echo ""
echo "============================================================"
echo " All seeds done."
echo " Results: runs/env_001/experiments/seed_{0..$((NUM_SEEDS-1))}/"
echo " Training: runs/env_001/training_runs/experiments/seed_{0..$((NUM_SEEDS-1))}/"
echo "============================================================"
