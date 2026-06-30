# Training Feedback

## External evaluation
- score: -142.797815
- episode_length: 88.100000 (mean)
- range: [-206.654021, -3.831011]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.011850 | 0.011850 | 0.010386 | 0.000000 | 1.981661 |
| progress_reward | 1.598827 | 1.671059 | 0.999991 | -3.217669 | 5.221412 |
| stability_penalty | -0.122471 | 0.122471 | 1.000000 | -0.516735 | -0.000000 |
| total_reward | 1.488206 | 1.571583 | 1.000000 | -3.422873 | 4.793724 |
| generated_reward | 1.488206 | 1.571583 | 1.000000 | -3.422873 | 4.793724 |
| original_env_reward | -1.538719 | 3.915060 | 1.000000 | -100.000000 | 156.650270 |

## Signals
early_failure_or_crash; sparse_proxy:landing_shaping; penalty_dominance:generated_reward; penalty_dominance:original_env_reward
