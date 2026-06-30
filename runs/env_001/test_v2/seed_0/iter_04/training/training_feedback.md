# Training Feedback

## External evaluation
- score: -114.009566
- episode_length: 74.100000 (mean)
- range: [-124.079149, -107.616409]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097182 | 0.097182 | 1.000000 | -0.169501 | -0.000110 |
| landing_quality | 0.010215 | 0.010215 | 0.012892 | 0.000000 | 1.893015 |
| progress_reward | 0.404316 | 0.427704 | 0.999997 | -1.011574 | 1.062317 |
| stability_penalty | -0.093721 | 0.093721 | 1.000000 | -1.047359 | -0.000000 |
| total_reward | 0.223628 | 0.299637 | 1.000000 | -1.487473 | 1.866444 |
| generated_reward | 0.223628 | 0.299637 | 1.000000 | -1.487473 | 1.866444 |
| original_env_reward | -1.590308 | 2.410149 | 1.000000 | -100.000000 | 135.446099 |

## Signals
early_failure_or_crash; sparse_proxy:landing_quality; penalty_dominance:original_env_reward
