#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/env001_deepseek_rag.yaml}
PREFIX=${2:-iter_exp}
TOTAL_TIMESTEPS=${3:-1000000}
EVAL_EPISODES=${4:-10}
MOCK_FLAG=${5:-}
ROUNDS=3

if [[ "$MOCK_FLAG" == "--mock" ]]; then
  MOCK="--mock"
else
  MOCK=""
fi

EXP_ROOT="runs/env_001/experiments/${PREFIX}"
MEMORY_PATH="runs/env_001/memory/reward_memory.md"

pad_iter() {
  printf "%02d" "$1"
}

echo "============================================================"
echo " Three-round iterative reward experiment"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "EXP_ROOT        : $EXP_ROOT"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo "MOCK            : ${MOCK_FLAG:-false}"
echo ""

PREVIOUS_REWARD=""

for ITER in $(seq 1 "$ROUNDS"); do
  ITER_PAD=$(pad_iter "$ITER")
  ITER_DIR="experiments/${PREFIX}/iter_${ITER_PAD}"
  GEN_RUN="${ITER_DIR}/generation"
  TRAIN_RUN="${ITER_DIR}/training"
  TRAIN_DIR="runs/env_001/training_runs/${TRAIN_RUN}"
  REWARD_VERSION="v${ITER}"

  echo "------------------------------------------------------------"
  echo " Iteration ${ITER}/${ROUNDS}"
  echo "------------------------------------------------------------"

  if [[ "$ITER" -eq 1 ]]; then
    echo "[iter ${ITER}] Generate initial reward_v1"
    python -m pipeline.run_direct_generation_pipeline \
      --config "$CONFIG" \
      --run-name "$GEN_RUN" \
      $MOCK

    CURRENT_REWARD="runs/env_001/${GEN_RUN}/reward_v1.py"
  else
    CONTEXT_PATH="${EXP_ROOT}/iter_${ITER_PAD}/iteration_context.md"

    echo "[iter ${ITER}] Build iteration context from previous training feedback"
    PREV_ITER=$((ITER - 1))
    PREV_PAD=$(pad_iter "$PREV_ITER")
    PREV_TRAIN_DIR="runs/env_001/training_runs/experiments/${PREFIX}/iter_${PREV_PAD}/training"
    python -m pipeline.run_04_build_iteration_context \
      --train-run-dir "$PREV_TRAIN_DIR" \
      --memory "$MEMORY_PATH" \
      --out "$CONTEXT_PATH"

    echo "[iter ${ITER}] Revise reward_${REWARD_VERSION}"
    python -m pipeline.run_05_reward_revision \
      --config "$CONFIG" \
      --previous-reward "$PREVIOUS_REWARD" \
      --iteration-context "$CONTEXT_PATH" \
      --out-run-name "$GEN_RUN" \
      --reward-version "$REWARD_VERSION" \
      $MOCK

    CURRENT_REWARD="runs/env_001/${GEN_RUN}/reward_${REWARD_VERSION}.py"
  fi

  echo "[iter ${ITER}] Train ${REWARD_VERSION}"
  python -m training.train_sb3_wrapper \
    --config "$CONFIG" \
    --reward "$CURRENT_REWARD" \
    --run-name "$TRAIN_RUN" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES"

  echo "[iter ${ITER}] Update compact reward memory"
  python -m pipeline.run_06_update_reward_memory \
    --iter "$ITER" \
    --train-run-dir "$TRAIN_DIR" \
    --memory "$MEMORY_PATH"

  PREVIOUS_REWARD="$CURRENT_REWARD"

done

echo ""
echo "============================================================"
echo "Done. Key files to inspect"
echo "============================================================"
for ITER in $(seq 1 "$ROUNDS"); do
  ITER_PAD=$(pad_iter "$ITER")
  ITER_DIR="${EXP_ROOT}/iter_${ITER_PAD}"
  REWARD_VERSION="v${ITER}"
  echo "iter ${ITER}:"
  echo "  reward     : ${ITER_DIR}/generation/reward_${REWARD_VERSION}.py"
  echo "  feedback   : runs/env_001/training_runs/experiments/${PREFIX}/iter_${ITER_PAD}/training/training_feedback.md"
  if [[ "$ITER" -gt 1 ]]; then
    echo "  context    : ${ITER_DIR}/iteration_context.md"
  fi
  echo ""
done
echo "memory: ${MEMORY_PATH}"
