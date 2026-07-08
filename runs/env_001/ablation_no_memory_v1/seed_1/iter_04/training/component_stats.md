# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.deceleration_bonus | 0.035070 | 0.035070 | 0.112360 | 0.312117 | 0.312117 | 0.000000 | 3.783314 | 1003520 |
| component.progress_delta | 0.047801 | 0.050642 | 0.999998 | 0.047801 | 0.050642 | -0.116445 | 0.126438 | 1003520 |
| component.stability_penalty | -0.066173 | 0.066173 | 1.000000 | -0.066173 | 0.066173 | -0.479439 | -0.000000 | 1003520 |
| component.total_reward | 0.016697 | 0.054907 | 1.000000 | 0.016697 | 0.054907 | -0.387600 | 3.772221 | 1003520 |
| generated_reward | 0.016697 | 0.054907 | 1.000000 | 0.016697 | 0.054907 | -0.387600 | 3.772221 | 1003520 |
| original_env_reward | -1.586114 | 2.419620 | 1.000000 | -1.586114 | 2.419620 | -100.000000 | 134.204995 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| deceleration_bonus | 2.483271 | 2.483271 | 0.081563 | 4.376348 | 14172 |
| progress_delta | 3.384297 | 3.384297 | 0.436263 | 4.213091 | 14172 |
| stability_penalty | -4.685219 | 4.685219 | -10.734469 | -3.888249 | 14172 |
| total_reward | 1.182349 | 1.716844 | -7.159909 | 3.973429 | 14172 |
