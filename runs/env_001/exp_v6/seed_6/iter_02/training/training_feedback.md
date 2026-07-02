# Training Feedback

## Training outcome
score=156.451604, len=916.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002529 | 0.002736 | 0.999713 | 1.000000 |
| soft_landing_proxy | 0.061126 | 0.061126 | 0.611263 | 24.166297 |
| stability_penalty | -0.000559 | 0.000559 | 1.000000 | -0.220901 |
| total_reward | 0.063097 | 0.063418 | 1.000000 | 24.945396 |
| generated_reward | 0.063097 | 0.063418 | 1.000000 | 24.945396 |
| original_env_reward | 0.039995 | 1.263796 | 1.000000 | 15.812157 |

## Distribution
- score: mean=156.451604, min=52.240116, max=199.098675
- episode_length: mean=916.200000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
