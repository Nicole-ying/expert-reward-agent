# Training Feedback

## Training outcome
score=191.825372, len=825.700000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.003214 | 0.003449 | 0.999750 | 1.000000 |
| soft_landing_proxy | 0.660904 | 0.660904 | 0.891088 | 205.660229 |
| stability_penalty | -0.000466 | 0.000466 | 0.999977 | -0.144982 |
| total_reward | 0.130805 | 0.132283 | 1.000000 | 40.704054 |
| generated_reward | 0.130805 | 0.132283 | 1.000000 | 40.704054 |
| original_env_reward | -0.121832 | 2.130995 | 1.000000 | -37.911712 |

## Distribution
- score: mean=191.825372, min=136.887288, max=267.195584
- episode_length: mean=825.700000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
