# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.003475 | 0.003789 | 0.997710 | 0.003483 | 0.003797 | -0.027977 | 0.036258 | 1003520 |
| component.soft_landing_proxy | 1.121948 | 1.121948 | 0.537436 | 2.087593 | 2.087593 | 0.000000 | 2.992025 | 1003520 |
| component.stability_penalty | -0.001468 | 0.001468 | 1.000000 | -0.001468 | 0.001468 | -0.058941 | -0.000000 | 1003520 |
| component.total_reward | 1.123955 | 1.124389 | 1.000000 | 1.123955 | 1.124389 | -0.078727 | 2.992024 | 1003520 |
| generated_reward | 1.123955 | 1.124389 | 1.000000 | 1.123955 | 1.124389 | -0.078727 | 2.992024 | 1003520 |
| original_env_reward | -0.033027 | 1.621370 | 1.000000 | -0.033027 | 1.621370 | -100.000000 | 132.824520 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 1.159237 | 1.159424 | -0.279762 | 1.418401 | 3004 |
| soft_landing_proxy | 374.093306 | 374.093306 | 0.000000 | 2442.853443 | 3004 |
| stability_penalty | -0.489957 | 0.489957 | -3.953360 | -0.083353 | 3004 |
| total_reward | 374.762586 | 374.778352 | -0.969523 | 2443.902013 | 3004 |
