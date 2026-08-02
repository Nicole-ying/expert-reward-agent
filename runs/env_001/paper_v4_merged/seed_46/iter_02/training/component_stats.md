# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_proximity_progress | 0.016029 | 0.016927 | 0.999995 | 0.016029 | 0.016927 | -0.040149 | 0.042039 | 1003520 |
| component.orientation_penalty | -0.000637 | 0.000637 | 0.012680 | -0.050237 | 0.050237 | -0.642103 | -0.000000 | 1003520 |
| component.terminal_success_bonus | 0.000857 | 0.000857 | 0.004286 | 0.200000 | 0.200000 | 0.000000 | 0.200000 | 1003520 |
| component.total_reward | 0.016249 | 0.018103 | 0.999995 | 0.016249 | 0.018103 | -0.642792 | 0.204561 | 1003520 |
| generated_reward | 0.016249 | 0.018103 | 0.999995 | 0.016249 | 0.018103 | -0.642792 | 0.204561 | 1003520 |
| original_env_reward | -1.572855 | 2.522949 | 1.000000 | -1.572855 | 2.522949 | -100.000000 | 138.185240 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_proximity_progress | 1.131438 | 1.131438 | 0.139714 | 1.395299 | 14216 |
| orientation_penalty | -0.044968 | 0.044968 | -13.813707 | 0.000000 | 14216 |
| terminal_success_bonus | 0.060509 | 0.060509 | 0.000000 | 0.200000 | 14216 |
| total_reward | 1.146979 | 1.210200 | -13.459150 | 1.580336 | 14216 |
