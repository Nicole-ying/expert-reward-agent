#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/env001_deepseek_rag.yaml}
PREFIX=${2:-iter_exp}
TOTAL_TIMESTEPS=${3:-1000000}
EVAL_EPISODES=${4:-10}
MOCK_FLAG=${5:-}

if [[ "$MOCK_FLAG" == "--mock" ]]; then
  MOCK="--mock"
else
  MOCK=""
fi

GEN1="${PREFIX}_gen_v1"
TRAIN1="${PREFIX}_train_v1"
ITER2="${PREFIX}_iter_v2"
GEN2="${PREFIX}_gen_v2"
TRAIN2="${PREFIX}_train_v2"
ITER3="${PREFIX}_iter_v3"
GEN3="${PREFIX}_gen_v3"
TRAIN3="${PREFIX}_train_v3"

echo "============================================================"
echo " Three-round reward experiment: v1 train -> v2 train -> v3 train"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo "MOCK            : ${MOCK_FLAG:-false}"
echo ""

echo "[1/9] Generate reward v1"
python -m pipeline.run_direct_generation_pipeline \
  --config "$CONFIG" \
  --run-name "$GEN1" \
  $MOCK

REWARD1="runs/env_001/${GEN1}/reward_v1.py"

echo "[2/9] Train reward v1"
python -m training.train_sb3_wrapper \
  --config "$CONFIG" \
  --reward "$REWARD1" \
  --run-name "$TRAIN1" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES"

echo "[3/9] Build iteration context for reward v2"
python -m pipeline.run_04_build_iteration_context \
  --train-run-dir "runs/env_001/training_runs/${TRAIN1}" \
  --out "runs/env_001/iterations/${ITER2}/iteration_context.md"

echo "[4/9] Revise reward v2"
python -m pipeline.run_05_reward_revision \
  --config "$CONFIG" \
  --previous-reward "$REWARD1" \
  --iteration-context "runs/env_001/iterations/${ITER2}/iteration_context.md" \
  --out-run-name "$GEN2" \
  --reward-version v2 \
  $MOCK

REWARD2="runs/env_001/${GEN2}/reward_v2.py"

echo "[5/9] Train reward v2"
python -m training.train_sb3_wrapper \
  --config "$CONFIG" \
  --reward "$REWARD2" \
  --run-name "$TRAIN2" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES"

echo "[6/9] Build iteration context for reward v3"
python -m pipeline.run_04_build_iteration_context \
  --train-run-dir "runs/env_001/training_runs/${TRAIN2}" \
  --out "runs/env_001/iterations/${ITER3}/iteration_context.md"

echo "[7/9] Revise reward v3"
python -m pipeline.run_05_reward_revision \
  --config "$CONFIG" \
  --previous-reward "$REWARD2" \
  --iteration-context "runs/env_001/iterations/${ITER3}/iteration_context.md" \
  --out-run-name "$GEN3" \
  --reward-version v3 \
  $MOCK

REWARD3="runs/env_001/${GEN3}/reward_v3.py"

echo "[8/9] Train reward v3"
python -m training.train_sb3_wrapper \
  --config "$CONFIG" \
  --reward "$REWARD3" \
  --run-name "$TRAIN3" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES"

echo "[9/9] Done"
echo ""
echo "Key files to inspect:"
echo "  v1 reward: runs/env_001/${GEN1}/reward_v1.py"
echo "  v1 feedback: runs/env_001/training_runs/${TRAIN1}/training_feedback.md"
echo "  v2 context: runs/env_001/iterations/${ITER2}/iteration_context.md"
echo "  v2 reward: runs/env_001/${GEN2}/reward_v2.py"
echo "  v2 feedback: runs/env_001/training_runs/${TRAIN2}/training_feedback.md"
echo "  v3 context: runs/env_001/iterations/${ITER3}/iteration_context.md"
echo "  v3 reward: runs/env_001/${GEN3}/reward_v3.py"
echo "  v3 feedback: runs/env_001/training_runs/${TRAIN3}/training_feedback.md"
