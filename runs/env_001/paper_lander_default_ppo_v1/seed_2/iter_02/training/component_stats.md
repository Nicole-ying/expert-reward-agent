# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.032265 | 0.034130 | 0.999992 | 0.032266 | 0.034130 | -0.081768 | 0.084406 | 1001472 |
| component.soft_landing_proxy | 0.004572 | 0.004572 | 0.004572 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1001472 |
| component.stability_penalty | -0.139528 | 0.139528 | 1.000000 | -0.139528 | 0.139528 | -1.003034 | -0.000000 | 1001472 |
| component.total_reward | -0.102690 | 0.111639 | 1.000000 | -0.102690 | 0.111639 | -1.026676 | 1.000386 | 1001472 |
| generated_reward | -0.102690 | 0.111639 | 1.000000 | -0.102690 | 0.111639 | -1.026676 | 1.000386 | 1001472 |
| original_env_reward | -1.534480 | 2.420943 | 1.000000 | -1.534480 | 2.420943 | -100.000000 | 95.961334 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 2.262411 | 2.262775 | -1.653825 | 2.800113 | 14282 |
| soft_landing_proxy | 0.320613 | 0.320613 | 0.000000 | 1.000000 | 14282 |
| stability_penalty | -9.783659 | 9.783659 | -32.656674 | -6.395816 | 14282 |
| total_reward | -7.200635 | 7.200635 | -30.121698 | -2.681650 | 14282 |
