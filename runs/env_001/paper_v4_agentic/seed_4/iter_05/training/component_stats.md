# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.153475 | 0.153475 | 0.767374 | -0.200000 | 0.200000 | -0.200000 | -0.000000 | 1003520 |
| component.landing_progress | 66.073010 | 66.073010 | 0.847982 | 77.917929 | 77.917929 | 0.000000 | 200.000000 | 1003520 |
| component.progress_gated | 0.398574 | 0.398574 | 0.699397 | 0.569883 | 0.569883 | 0.000000 | 1.991380 | 1003520 |
| component.total_reward | 6.900781 | 6.965483 | 0.966315 | 7.141340 | 7.208297 | -0.200000 | 20.000000 | 1003520 |
| generated_reward | 6.900781 | 6.965483 | 0.966315 | 7.141340 | 7.208297 | -0.200000 | 20.000000 | 1003520 |
| original_env_reward | 0.015136 | 1.907415 | 1.000000 | 0.015136 | 1.907415 | -100.000000 | 126.123970 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -62.615554 | 62.615554 | -167.800000 | -5.800000 | 2456 |
| landing_progress | 26941.671478 | 26941.671478 | 0.000000 | 167202.904141 | 2456 |
| progress_gated | 162.538592 | 162.538592 | 0.000000 | 390.686012 | 2456 |
| total_reward | 2813.813453 | 2815.014237 | -89.312609 | 16784.407743 | 2456 |
