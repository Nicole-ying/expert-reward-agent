# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.004859 | 0.004859 | 0.242937 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.goal_proximity | -0.972045 | 0.972045 | 1.000000 | -0.972045 | 0.972045 | -1.695007 | -0.000277 | 1003520 |
| component.orientation_penalty | -0.003285 | 0.003285 | 0.999968 | -0.003286 | 0.003286 | -2.133270 | -0.000000 | 1003520 |
| component.safe_landing | 0.045170 | 0.045170 | 0.015251 | 2.961741 | 2.961741 | 0.000000 | 10.000000 | 1003520 |
| component.total_reward | -0.935018 | 1.021363 | 1.000000 | -0.935018 | 1.021363 | -3.086138 | 9.957250 | 1003520 |
| generated_reward | -0.935018 | 1.021363 | 1.000000 | -0.935018 | 1.021363 | -3.086138 | 9.957250 | 1003520 |
| original_env_reward | -1.679850 | 2.509905 | 1.000000 | -1.679850 | 2.509905 | -100.000000 | 141.927859 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -0.340867 | 0.340867 | -2.140000 | -0.020000 | 14303 |
| goal_proximity | -68.183620 | 68.183620 | -175.556108 | -40.600459 | 14303 |
| orientation_penalty | -0.230503 | 0.230503 | -55.625356 | -0.000028 | 14303 |
| safe_landing | 3.169227 | 3.169227 | 0.000000 | 10.618232 | 14303 |
| total_reward | -65.585763 | 65.585763 | -206.121912 | -30.539266 | 14303 |
