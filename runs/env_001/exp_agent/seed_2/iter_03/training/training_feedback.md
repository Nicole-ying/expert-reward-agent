# Training Feedback

## Training outcome
score=-111.045073, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.008878 | 0.008878 | 0.177566 | -0.055402 |
| progress_delta_reward | 0.160253 | 0.169687 | 0.999995 | 1.000000 |
| soft_landing_bonus | 0.033280 | 0.033280 | 0.017958 | 0.207670 |
| stability_penalty | -0.039070 | 0.039070 | 1.000000 | -0.243800 |
| total_reward | 0.145585 | 0.168381 | 1.000000 | 0.908468 |
| generated_reward | 0.145585 | 0.168381 | 1.000000 | 0.908468 |
| original_env_reward | -1.566098 | 2.350723 | 1.000000 | -9.772639 |

## Distribution
- score: mean=-111.045073, min=-124.631492, max=-97.710495
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
