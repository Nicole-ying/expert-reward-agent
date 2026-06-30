# Training Feedback

## External evaluation
- score: -108.448347
- episode_length: 74.100000 (mean)
- range: [-120.984102, -97.182870]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| distance_anchor | -0.097128 | 0.097128 | 1.000000 | -0.169501 | -0.000119 |
| progress_reward | 0.242793 | 0.256757 | 0.999993 | -0.621055 | 0.636037 |
| soft_landing_bonus | 0.004115 | 0.004115 | 0.009211 | 0.000000 | 0.914699 |
| stability_penalty | -0.216834 | 0.216834 | 1.000000 | -1.739710 | -0.000000 |
| total_reward | -0.067054 | 0.117649 | 1.000000 | -1.895030 | 0.912603 |
| generated_reward | -0.067054 | 0.117649 | 1.000000 | -1.895030 | 0.912603 |
| original_env_reward | -1.570631 | 2.392677 | 1.000000 | -100.000000 | 136.063634 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:original_env_reward
