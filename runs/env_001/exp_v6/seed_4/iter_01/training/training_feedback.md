# Training Feedback

## Training outcome
score=-111.441730, len=71.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.016054 | 0.016993 | 0.999993 | 0.016054 |
| soft_landing_proxy | 0.000777 | 0.000777 | 0.003883 | 0.000777 |
| stability_penalty | -0.057148 | 0.057148 | 1.000000 | -0.057148 |
| total_reward | -0.040317 | 0.041800 | 1.000000 | -0.040317 |
| generated_reward | -0.040317 | 0.041800 | 1.000000 | -0.040317 |
| original_env_reward | -1.548622 | 2.378699 | 1.000000 | -1.548622 |

## Distribution
- score: mean=-111.441730, min=-122.161769, max=-95.059093
- episode_length: mean=71.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
