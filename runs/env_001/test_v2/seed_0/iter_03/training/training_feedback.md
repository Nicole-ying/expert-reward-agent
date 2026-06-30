# Training Feedback

## External evaluation
- score: -111.238907
- episode_length: 74.100000 (mean)
- range: [-120.969771, -105.485038]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.096984 | 0.096984 | 1.000000 | -0.170105 | -0.000176 |
| progress_reward | 0.323339 | 0.341766 | 0.999995 | -0.807268 | 0.844599 |
| soft_landing_bonus | 0.003982 | 0.003982 | 0.009369 | 0.000000 | 0.914458 |
| stability_penalty | -0.218167 | 0.218167 | 1.000000 | -1.570842 | -0.000001 |
| total_reward | 0.012169 | 0.155207 | 1.000000 | -2.001005 | 0.919099 |
| generated_reward | 0.012169 | 0.155207 | 1.000000 | -2.001005 | 0.919099 |
| original_env_reward | -1.526222 | 2.404256 | 1.000000 | -100.000000 | 129.532942 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:original_env_reward
