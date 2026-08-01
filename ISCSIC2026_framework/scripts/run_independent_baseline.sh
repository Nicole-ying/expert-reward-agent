#!/usr/bin/env bash
# Same initial-generation path and PPO budget as CREATE, but no memory or repair.
set -euo pipefail

python scripts/run_independent_reward_baseline.py \
  --config configs/env001_paper_v4.yaml \
  --prefix independent_prompt_matched_v4 \
  --candidates 10 \
  --start-seed 0 \
  --num-seeds 5 \
  --total-timesteps 1000000 \
  --eval-episodes 20
