# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.032254 | 0.034114 | 0.999996 | 0.032254 | 0.034115 | -0.081198 | 0.084518 | 1001472 |
| component.soft_landing_proxy | 0.004221 | 0.004221 | 0.004221 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1001472 |
| component.stability_penalty | -0.011409 | 0.011409 | 1.000000 | -0.011409 | 0.011409 | -0.082557 | -0.000000 | 1001472 |
| component.total_reward | 0.025066 | 0.029495 | 1.000000 | 0.025066 | 0.029495 | -0.135154 | 1.000085 | 1001472 |
| generated_reward | 0.025066 | 0.029495 | 1.000000 | 0.025066 | 0.029495 | -0.135154 | 1.000085 | 1001472 |
| original_env_reward | -1.465070 | 2.504427 | 1.000000 | -1.465070 | 2.504427 | -100.000000 | 116.953285 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 2.261698 | 2.261730 | -0.231535 | 2.783208 | 14282 |
| soft_landing_proxy | 0.295967 | 0.295967 | 0.000000 | 1.000000 | 14282 |
| stability_penalty | -0.799991 | 0.799991 | -2.786622 | -0.515634 | 14282 |
| total_reward | 1.757673 | 1.761610 | -2.001090 | 3.210888 | 14282 |
