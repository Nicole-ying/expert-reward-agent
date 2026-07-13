# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.001015 | 0.001015 | 0.999905 | -0.001016 | 0.001016 | -1.375617 | -0.000000 | 1003520 |
| component.angular_vel_penalty | -0.000791 | 0.000791 | 0.996779 | -0.000794 | 0.000794 | -0.563129 | -0.000000 | 1003520 |
| component.gated_progress | 0.007228 | 0.007887 | 0.999999 | 0.007228 | 0.007887 | -0.026227 | 0.020365 | 1003520 |
| component.total_reward | 0.005421 | 0.009157 | 1.000000 | 0.005421 | 0.009157 | -1.378532 | 0.020194 | 1003520 |
| generated_reward | 0.005421 | 0.009157 | 1.000000 | 0.005421 | 0.009157 | -1.378532 | 0.020194 | 1003520 |
| original_env_reward | -0.412812 | 2.897418 | 1.000000 | -0.412812 | 2.897418 | -100.000000 | 129.292165 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.099612 | 0.099612 | -18.155521 | -0.000113 | 10230 |
| angular_vel_penalty | -0.077624 | 0.077624 | -1.517309 | -0.000428 | 10230 |
| gated_progress | 0.708948 | 0.708963 | -0.075613 | 1.087345 | 10230 |
| total_reward | 0.531712 | 0.682508 | -18.386273 | 1.077497 | 10230 |
