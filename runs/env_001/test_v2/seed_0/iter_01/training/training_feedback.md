# Training Feedback

## External evaluation
- score: -110.396465
- episode_length: 74.100000 (mean)
- range: [-121.667740, -97.714115]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097176 | 0.097176 | 1.000000 | -0.169501 | -0.000123 |
| progress_reward | 0.160700 | 0.170018 | 0.999987 | -0.416105 | 0.424384 |
| soft_landing_bonus | 0.009905 | 0.009905 | 0.004953 | 0.000000 | 2.000000 |
| stability_penalty | -0.342913 | 0.342913 | 1.000000 | -2.804435 | -0.000000 |
| total_reward | -0.269484 | 0.288008 | 1.000000 | -3.049366 | 1.996958 |
| generated_reward | -0.269484 | 0.288008 | 1.000000 | -3.049366 | 1.996958 |
| original_env_reward | -1.562164 | 2.397884 | 1.000000 | -100.000000 | 131.099741 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
