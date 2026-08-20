# Training Feedback

## Training outcome
score=66.522094, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.045367 | 0.048510 | 0.999966 | 1.000000 |
| soft_landing_proxy | 0.077214 | 0.077214 | 0.077214 | 1.701983 |
| stability_penalty | -0.000628 | 0.000628 | 1.000000 | -0.013850 |
| total_reward | 0.121953 | 0.125075 | 1.000000 | 2.688133 |
| generated_reward | 0.121953 | 0.125075 | 1.000000 | 2.688133 |
| original_env_reward | -0.250546 | 4.386100 | 1.000000 | -5.522627 |

## Distribution
- score: mean=66.522094, min=40.938003, max=95.685433
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
