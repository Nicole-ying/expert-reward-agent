# Training Feedback

## Training outcome
score=-112.368158, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.016123 | 0.017044 | 1.000000 | 1.000000 |
| soft_landing_proxy | 0.000986 | 0.000986 | 0.001971 | 0.061126 |
| stability_penalty | -0.014204 | 0.014204 | 1.000000 | -0.880965 |
| total_reward | 0.002905 | 0.008829 | 1.000000 | 0.180160 |
| generated_reward | 0.002905 | 0.008829 | 1.000000 | 0.180160 |
| original_env_reward | -1.588476 | 2.430089 | 1.000000 | -98.521943 |

## Distribution
- score: mean=-112.368158, min=-118.802286, max=-101.820135
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
