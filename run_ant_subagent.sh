#!/usr/bin/env bash
set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
source /home/utseus22/anaconda3/etc/profile.d/conda.sh
conda activate stable-eureka
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

CONFIG="configs/env005_paper_v4_agentic.yaml"
PREFIX="paper_ant_subagent_20260728"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20
LOG_FILE="logs/${PREFIX}.log"

echo "============================================================"
echo " Ant Subagent Experiment"
echo "  Config : env005_paper_v4_agentic (extends env005_paper_v4)"
echo "  Conda  : stable-eureka"
echo "  LLM    : DeepSeek v4-pro"
echo "  Subagent Investigator: enabled, max_turns=3"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : 5"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "EVAL_EPISODES   : $EVAL_EPISODES"
echo "LOG             : $LOG_FILE"
echo ""

mkdir -p logs

# The multi_seed runner loops seeds 0-4 internally, so one call covers all 5.
python -m pipeline.run_multi_seed_experiment \
  --config "$CONFIG" \
  --prefix "$PREFIX" \
  --rounds "$ROUNDS" \
  --total-timesteps "$TOTAL_TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES"
