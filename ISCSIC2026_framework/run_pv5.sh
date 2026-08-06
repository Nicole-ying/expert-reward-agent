#!/usr/bin/env bash
set -euo pipefail

# ── env vars for API access ──
export SSL_CERT_FILE="C:/ProgramData/miniconda3/envs/eure/Lib/site-packages/certifi/cacert.pem"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"

CONFIG="configs/env001_paper_v5.yaml"
PREFIX="pv5"
ROUNDS=10
TOTAL_TIMESTEPS=1000000
EVAL_EPISODES=20
FROZEN_CONTEXT="runs/env_001/pv5_frozen_context"
SEEDS=(0 1 2 3 4)

LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "============================================================"
echo " pv5 Sequential Experiment"
echo "============================================================"
echo "CONFIG          : $CONFIG"
echo "PREFIX          : $PREFIX"
echo "SEEDS           : ${SEEDS[*]}"
echo "ROUNDS          : $ROUNDS"
echo "TOTAL_TIMESTEPS : $TOTAL_TIMESTEPS"
echo "START TIME      : $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

TOTAL=${#SEEDS[@]}
for i in "${!SEEDS[@]}"; do
  seed=${SEEDS[$i]}
  LOGFILE="$LOGDIR/${PREFIX}_seed${seed}.log"
  order=$((i + 1))

  echo "############################################################"
  echo " SEED $seed  ($order / $TOTAL)  —  $(date '+%H:%M:%S')"
  echo " log: $LOGFILE"
  echo "############################################################"

  python -u -m pipeline.run_iterative_experiment \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --seed "$seed" \
    --rounds "$ROUNDS" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --frozen-context-from "$FROZEN_CONTEXT" \
    > "$LOGFILE" 2>&1

  echo ""
  echo " SEED $seed DONE  —  $(date '+%H:%M:%S')"
  echo ""
done

echo "============================================================"
echo " All seeds done.  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
