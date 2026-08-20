# Training Feedback

## Training outcome
score=-112.636699, len=70.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_reward | -0.972025 | 0.972025 | 1.000000 | -0.972025 |
| landing_bonus | 0.009277 | 0.009277 | 0.001855 | 0.009277 |
| stability_penalty | -0.144418 | 0.144418 | 1.000000 | -0.144418 |
| total_reward | -1.107166 | 1.125475 | 1.000000 | -1.107166 |
| generated_reward | -1.107166 | 1.125475 | 1.000000 | -1.107166 |
| original_env_reward | -1.632276 | 2.447112 | 1.000000 | -1.632276 |

## Distribution
- score: mean=-112.636699, min=-125.170388, max=-95.059093
- episode_length: mean=70.500000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
