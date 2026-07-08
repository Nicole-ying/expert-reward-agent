# Training Feedback

## Training outcome
score=249.016017, len=276.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002800 | 0.003082 | 0.999744 | 1.000000 |
| soft_landing_proxy | 0.205860 | 0.205860 | 0.697251 | 73.523114 |
| stability_penalty | -0.000707 | 0.000707 | 1.000000 | -0.252563 |
| total_reward | 0.207952 | 0.208327 | 1.000000 | 74.270551 |
| generated_reward | 0.207952 | 0.208327 | 1.000000 | 74.270551 |
| original_env_reward | -0.006390 | 1.578897 | 1.000000 | -2.282318 |

## Distribution
- score: mean=249.016017, min=11.977674, max=305.768457
- episode_length: mean=276.100000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
