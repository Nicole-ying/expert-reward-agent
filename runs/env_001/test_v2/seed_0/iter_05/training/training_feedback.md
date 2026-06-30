# Training Feedback

## External evaluation
- score: -111.139031
- episode_length: 74.100000 (mean)
- range: [-122.534771, -105.145711]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097016 | 0.097016 | 1.000000 | -0.169501 | -0.000019 |
| progress_delta_reward | 0.160916 | 0.170245 | 0.999990 | -0.418539 | 0.425586 |
| soft_landing_bonus | 0.009273 | 0.009273 | 0.004637 | 0.000000 | 2.000000 |
| stability_penalty | -0.340556 | 0.340556 | 1.000000 | -2.804435 | -0.000000 |
| total_reward | -0.267383 | 0.284934 | 1.000000 | -3.049366 | 1.993564 |
| generated_reward | -0.267383 | 0.284934 | 1.000000 | -3.049366 | 1.993564 |
| original_env_reward | -1.538081 | 2.403107 | 1.000000 | -100.000000 | 126.195419 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus
