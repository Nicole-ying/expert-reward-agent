#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/env001_deepseek_rag.yaml}
GEN_RUN=${2:-deepseek_full_run_001}
TRAIN_RUN=${3:-ppo_full_run_001}
TOTAL_TIMESTEPS=${4:-10000}
EVAL_EPISODES=${5:-5}
MOCK_FLAG=${6:-}

if [[ "$MOCK_FLAG" == "--mock" ]]; then
  GENERATION_MOCK="--mock"
else
  GENERATION_MOCK=""
fi

echo "============================================================"
echo " Full experiment: reward generation -> PPO training -> eval"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "GEN_RUN         : $GEN_RUN"
echo "TRAIN_RUN       : $TRAIN_RUN"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo "MOCK            : ${MOCK_FLAG:-false}"
echo ""

echo "[1/4] Generate environment_card.md, expert_reward_context.md, reward_v1.py"
python -m pipeline.run_direct_generation_pipeline \
  --config "$CONFIG" \
  --run-name "$GEN_RUN" \
  $GENERATION_MOCK

REWARD_PATH="runs/env_001/${GEN_RUN}/reward_v1.py"
VALIDATION_PATH="runs/env_001/${GEN_RUN}/validations/reward_v1.validation.json"

if [[ ! -f "$REWARD_PATH" ]]; then
  echo "ERROR: reward file not found: $REWARD_PATH"
  exit 1
fi

if [[ ! -f "$VALIDATION_PATH" ]]; then
  echo "ERROR: validation file not found: $VALIDATION_PATH"
  exit 1
fi

echo ""
echo "[2/4] Check reward validation"
python - <<PY
import json
from pathlib import Path
p = Path("$VALIDATION_PATH")
data = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps(data, ensure_ascii=False, indent=2))
if not data.get("valid", False):
    raise SystemExit("reward_v1 validation failed")
PY

echo ""
echo "[3/4] Train PPO with generated reward"
python -m training.train_sb3_wrapper \
  --config "$CONFIG" \
  --reward "$REWARD_PATH" \
  --run-name "$TRAIN_RUN" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES"

echo ""
echo "[4/4] Done"
echo "Generation outputs: runs/env_001/${GEN_RUN}/"
echo "Training outputs  : runs/env_001/training_runs/${TRAIN_RUN}/"
echo "Key files:"
echo "  - runs/env_001/${GEN_RUN}/reward_v1.py"
echo "  - runs/env_001/${GEN_RUN}/reward_v1.md"
echo "  - runs/env_001/${GEN_RUN}/validations/reward_v1.validation.json"
echo "  - runs/env_001/training_runs/${TRAIN_RUN}/eval_result.json"
echo "  - runs/env_001/training_runs/${TRAIN_RUN}/training_summary.json"
