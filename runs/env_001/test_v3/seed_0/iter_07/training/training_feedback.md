# Training Feedback

## External evaluation
- score: 146.572765
- episode_length: 564.600000 (mean)
- range: [-69.127123, 295.110743]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| progress_reward | 0.091198 | 0.100075 | 0.999585 | -0.354730 | 0.384918 |
| soft_landing_bonus | 0.396895 | 0.396895 | 0.198447 | 0.000000 | 2.000000 |
| stability_penalty | -0.145633 | 0.145633 | 1.000000 | -8.460213 | -0.000000 |
| total_reward | 0.342459 | 0.452185 | 1.000000 | -8.304337 | 2.004369 |
| generated_reward | 0.342459 | 0.452185 | 1.000000 | -8.304337 | 2.004369 |
| original_env_reward | -0.388388 | 2.619071 | 1.000000 | -100.000000 | 143.501626 |

## Signals
partial_progress; penalty_dominance:soft_landing_bonus; penalty_dominance:stability_penalty; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
