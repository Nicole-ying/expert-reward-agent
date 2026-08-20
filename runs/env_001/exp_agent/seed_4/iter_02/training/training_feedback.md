# Training Feedback

## Training outcome
score=149.939475, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.001851 | 0.002116 | 0.999649 | 1.000000 |
| soft_landing_proxy | 0.234186 | 0.234186 | 0.992302 | 126.523214 |
| stability_penalty | -0.002446 | 0.002446 | 1.000000 | -1.321567 |
| total_reward | 0.233590 | 0.233799 | 1.000000 | 126.201647 |
| generated_reward | 0.233590 | 0.233799 | 1.000000 | 126.201647 |
| original_env_reward | 0.052540 | 1.786705 | 1.000000 | 28.385489 |

## Distribution
- score: mean=149.939475, min=120.080880, max=176.487684
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
