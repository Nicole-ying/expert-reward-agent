# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_approach_progress | 0.031351 | 0.037104 | 0.999695 | 0.031360 | 0.037116 | -0.306370 | 0.384919 | 1003520 |
| component.orientation_stability | -0.071473 | 0.071473 | 0.999977 | -0.071475 | 0.071475 | -27.633504 | -0.000000 | 1003520 |
| component.terminal_soft_landing_bonus | 14.147451 | 14.147451 | 0.282949 | 50.000000 | 50.000000 | 0.000000 | 50.000000 | 1003520 |
| component.total_reward | 5.421546 | 5.896493 | 1.000000 | 5.421546 | 5.896493 | -20.000000 | 20.000000 | 1003520 |
| component.velocity_brake_near_target | -0.200210 | 0.200210 | 0.999835 | -0.200243 | 0.200243 | -13.435960 | -0.000000 | 1003520 |
| generated_reward | 5.421546 | 5.896493 | 1.000000 | 5.421546 | 5.896493 | -20.000000 | 20.000000 | 1003520 |
| original_env_reward | -0.090436 | 1.854115 | 1.000000 | -0.090436 | 1.854115 | -100.000000 | 123.582319 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_approach_progress | 10.607596 | 10.651850 | -11.529023 | 14.168174 | 2963 |
| orientation_stability | -24.187337 | 24.187337 | -273.393424 | -0.078736 | 2963 |
| terminal_soft_landing_bonus | 4787.073912 | 4787.073912 | 0.000000 | 39400.000000 | 2963 |
| total_reward | 1834.461230 | 1962.818832 | -637.711130 | 15671.196976 | 2963 |
| velocity_brake_near_target | -67.769660 | 67.769660 | -425.535241 | -20.359328 | 2963 |
