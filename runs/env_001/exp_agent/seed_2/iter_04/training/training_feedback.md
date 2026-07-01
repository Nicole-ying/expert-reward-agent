# Training Feedback

## Training outcome
score=-108.982652, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.015580 | 0.015580 | 0.311597 | -0.097896 |
| progress_delta_reward | 0.159147 | 0.168636 | 0.999993 | 1.000000 |
| soft_landing_bonus | 0.085496 | 0.085496 | 0.017304 | 0.537213 |
| stability_penalty | -0.270811 | 0.270811 | 1.000000 | -1.701641 |
| total_reward | -0.041748 | 0.205102 | 0.999999 | -0.262323 |
| generated_reward | -0.041748 | 0.205102 | 0.999999 | -0.262323 |
| original_env_reward | -1.506394 | 2.371567 | 1.000000 | -9.465418 |

## Distribution
- score: mean=-108.982652, min=-124.079149, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
