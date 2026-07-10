#!/bin/bash
# Paper baseline: run N training seeds, save each to a separate directory
# Usage: bash baselines/run_baseline_seeds.sh baselines/config_rlzoo.yaml 0 5
#        bash baselines/run_baseline_seeds.sh baselines/config_openai.yaml 1 5

set -e
CONFIG=${1:?Usage: $0 <config.yaml> <gpu_id> [num_seeds]}
GPU=${2:?GPU ID required}
NSEEDS=${3:-5}
REWARD=baselines/passthrough_reward.py
PREFIX=$(basename "$CONFIG" .yaml)
LOGDIR=logs/baselines
SAVEDIR=runs/baselines/${PREFIX}

mkdir -p "$LOGDIR" "$SAVEDIR"

echo "=== Baseline: $PREFIX | GPU $GPU | $NSEEDS seeds | save: $SAVEDIR ==="

declare -a SCORES

for s in $(seq 0 $((NSEEDS - 1))); do
    SEED_DIR="${SAVEDIR}/seed_${s}"
    echo "[$(date '+%H:%M:%S')] seed=$s ..."
    CUDA_VISIBLE_DEVICES=$GPU python -u -m training.train_sb3_wrapper \
        --config "$CONFIG" \
        --reward "$REWARD" \
        --seed "$s" \
        --save-dir "$SEED_DIR" \
        > "${LOGDIR}/${PREFIX}_seed${s}.log" 2>&1
    SCORE=$(python3 -c "import json; d=json.load(open('${SEED_DIR}/eval_result.json')); print(d['mean_eval_reward'])")
    SCORES+=("$SCORE")
    echo "[$(date '+%H:%M:%S')] seed=$s  score=$SCORE"
done

# Aggregate
echo ""
echo "=== Aggregate: $PREFIX ==="
printf '%s\n' "${SCORES[@]}" | python3 -c "
import sys, math
vals = [float(x) for x in sys.stdin.read().strip().split()]
mean = sum(vals)/len(vals)
std = math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1)) if len(vals)>1 else 0.0
print(f'  seeds: {vals}')
print(f'  mean ± std: {mean:.1f} ± {std:.1f}')
print(f'  min: {min(vals):.1f}  max: {max(vals):.1f}')
"
