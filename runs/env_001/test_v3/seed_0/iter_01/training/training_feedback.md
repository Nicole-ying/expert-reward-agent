# Training Feedback

## External evaluation
- score: -111.262458
- episode_length: 74.100000 (mean)
- range: [-123.713417, -96.378727]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.160497 | 0.169886 | 0.999990 | -0.420961 | 0.423601 |
| soft_landing_bonus | 0.010589 | 0.010589 | 0.005294 | 0.000000 | 2.000000 |
| stability_penalty | -0.339930 | 0.339930 | 1.000000 | -2.804435 | -0.000001 |
| total_reward | -0.168845 | 0.188854 | 1.000000 | -2.964972 | 2.006727 |
| generated_reward | -0.168845 | 0.188854 | 1.000000 | -2.964972 | 2.006727 |
| original_env_reward | -1.556501 | 2.357025 | 1.000000 | -100.000000 | 112.994905 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
