# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.030661 | 0.030661 | 0.999999 | -0.030661 | 0.030661 | -7.864957 | -0.000000 | 1003520 |
| component.fuel_penalty | -0.016360 | 0.016360 | 0.109068 | -0.150000 | 0.150000 | -0.150000 | 0.000000 | 1003520 |
| component.proximity_progress | 0.645582 | 0.682258 | 0.999999 | 0.645583 | 0.682259 | -1.689871 | 1.697815 | 1003520 |
| component.soft_landing_bonus | 0.020173 | 0.020173 | 0.015885 | 1.269933 | 1.269933 | 0.000000 | 3.999985 | 1003520 |
| component.total_reward | 0.618734 | 0.705205 | 1.000000 | 0.618734 | 0.705205 | -8.484289 | 3.988801 | 1003520 |
| generated_reward | 0.618734 | 0.705205 | 1.000000 | 0.618734 | 0.705205 | -8.484289 | 3.988801 | 1003520 |
| original_env_reward | -1.649823 | 2.370213 | 1.000000 | -1.649823 | 2.370213 | -100.000000 | 132.539326 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -2.148668 | 2.148668 | -113.746803 | -0.000003 | 14320 |
| fuel_penalty | -1.146432 | 1.146432 | -19.950000 | 0.000000 | 14320 |
| proximity_progress | 45.237007 | 45.237007 | 13.187629 | 56.006056 | 14320 |
| soft_landing_bonus | 1.413688 | 1.413688 | 0.000000 | 4.831310 | 14320 |
| total_reward | 43.355595 | 43.538917 | -109.202346 | 59.511458 | 14320 |
