#!/bin/bash
# 等 iter_05 训练跑完 → 更新 memory → 继续 iter_06+ → seeds 5-9
set -e

PROJECT_DIR="c:/Users/Administrator/Downloads/expert_eureka_env001_bridge_v9_direct_generator/expert_eureka_env001_bridge_v9_direct_generator"
source /c/ProgramData/miniconda3/etc/profile.d/conda.sh
conda activate eure
cd "$PROJECT_DIR"

CONFIG="configs/env001_deepseek_rag.yaml"
PREFIX="exp_v6"
MEMORY="runs/env_001/exp_v6/seed_4/memory/reward_memory.md"
TRAIN_DIR="runs/env_001/exp_v6/seed_4/iter_05/training"
SUMMARY="$TRAIN_DIR/training_summary.json"

echo "[$(date)] Waiting for iter_05 training to finish..."
while [ ! -f "$SUMMARY" ]; do
    sleep 10
done
echo "[$(date)] Training done!"

# 读取 eval 分数
SCORE=$(python -c "import json; d=json.load(open('$SUMMARY','r',encoding='utf-8')); print(d['external_eval']['mean_eval_reward'])")
echo "Score: $SCORE"

# 从 memory 读取历史最佳
BEST_SCORE=$(python -c "
import re, json
mem=open('$MEMORY','r',encoding='utf-8').read()
best=-9999
for line in mem.splitlines():
    if line.startswith('|') and not 'iter' in line and not '---' in line:
        cols=[c.strip() for c in line.split('|')]
        if len(cols)>=4:
            try: best=max(best,float(cols[3]))
            except: pass
print(best if best!=-9999 else '$SCORE')
")
echo "Best score from memory: $BEST_SCORE"

# 判断 decision
BETTER=$(python -c "print(1 if $SCORE > $BEST_SCORE + 5.0 else 0)")
TARGET=200.0
if [ "$BETTER" = "1" ]; then
    DECISION="new_best"
    BEST_ITER=5
    NEW_BEST=$SCORE
else
    DECISION="no_meaningful_improvement"
    BEST_ITER=$(python -c "
import re
mem=open('$MEMORY','r',encoding='utf-8').read()
best_iter=1; best_score=-9999
for line in mem.splitlines():
    if line.startswith('|') and not 'iter' in line and not '---' in line:
        cols=[c.strip() for c in line.split('|')]
        if len(cols)>=4:
            try:
                s=float(cols[3])
                if s>best_score: best_score=s; best_iter=int(cols[1])
            except: pass
print(best_iter)
")
    NEW_BEST=$BEST_SCORE
fi
echo "Decision: $DECISION (best_iter=$BEST_ITER, best_score=$NEW_BEST)"

# Step 1: 更新 memory
echo "[$(date)] Updating memory for iter_05..."
python -m pipeline.run_06_update_reward_memory \
    --iter 5 \
    --train-run-dir "$TRAIN_DIR" \
    --memory "$MEMORY" \
    --target-score $TARGET \
    --best-score "$NEW_BEST" \
    --best-iter "$BEST_ITER" \
    --decision "$DECISION"

# Step 2: 续跑 seed 4 从 iter 6
echo "[$(date)] Resuming seed 4 from iter 6..."
python -m pipeline.run_iterative_experiment \
    --config "$CONFIG" \
    --prefix "$PREFIX" \
    --seed 4 \
    --resume-from 6

echo "[$(date)] Seed 4 done!"

# Step 3: 跑剩余 seeds 5-9
for s in 5 6 7 8 9; do
    echo "[$(date)] ======== Starting seed $s ========"
    python -m pipeline.run_iterative_experiment \
        --config "$CONFIG" \
        --prefix "$PREFIX" \
        --seed "$s"
done

echo "[$(date)] ALL DONE!"
