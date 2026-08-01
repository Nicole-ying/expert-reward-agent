#!/usr/bin/env bash
# Five-seed native-reward PPO reference with the paper-v4 training budget.
set -euo pipefail

CONFIG="configs/env001_deepseek_rag.yaml"
REWARD="baselines/official_reward.py"

for seed in 0 1 2 3 4; do
  echo "=== Seed $seed ==="
  python -m training.train_sb3_wrapper \
    --config "$CONFIG" \
    --reward "$REWARD" \
    --run-name "official_baseline/seed_${seed}" \
    --save-dir "runs/env_001/official_baseline/seed_${seed}" \
    --total-timesteps 1000000 \
    --eval-episodes 20 \
    --seed "$seed"
done

echo "All seeds done."
