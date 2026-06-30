# Training Feedback

## External evaluation
- score: -111.792476
- episode_length: 74.100000 (mean)
- range: [-121.129284, -96.806702]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.241773 | 0.255777 | 0.999998 | -0.608839 | 0.634183 |
| soft_landing_bonus | 0.010212 | 0.010212 | 0.005106 | 0.000000 | 2.000000 |
| stability_penalty | -0.330620 | 0.330620 | 1.000000 | -1.469923 | -0.000000 |
| total_reward | -0.078635 | 0.110697 | 1.000000 | -1.813497 | 2.002875 |
| generated_reward | -0.078635 | 0.110697 | 1.000000 | -1.813497 | 2.002875 |
| original_env_reward | -1.586450 | 2.399495 | 1.000000 | -100.000000 | 139.587203 |

## Signals
early_failure_or_crash; sparse_proxy:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:original_env_reward
