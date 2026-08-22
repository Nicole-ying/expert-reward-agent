# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.002054 | 0.002054 | 0.005407 | -0.379941 | 0.379941 | -2.313901 | -0.000000 | 1015808 |
| component.energy_penalty | -0.021603 | 0.021603 | 1.000000 | -0.021603 | 0.021603 | -0.040000 | -0.000010 | 1015808 |
| component.gated_forward_reward | 0.558453 | 0.562320 | 1.000000 | 0.558453 | 0.562320 | -0.510315 | 1.500721 | 1015808 |
| component.total_reward | 0.534795 | 0.545268 | 1.000000 | 0.534795 | 0.545268 | -2.360240 | 1.473486 | 1015808 |
| generated_reward | 0.534795 | 0.545268 | 1.000000 | 0.534795 | 0.545268 | -2.360240 | 1.473486 | 1015808 |
| original_env_reward | 0.018841 | 0.332314 | 1.000000 | 0.018841 | 0.332314 | -100.000000 | 0.788929 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -1.166369 | 1.166369 | -94.469662 | 0.000000 | 1789 |
| energy_penalty | -12.216675 | 12.216675 | -34.913337 | -0.753429 | 1789 |
| gated_forward_reward | 314.920224 | 315.174924 | -14.640451 | 938.050903 | 1789 |
| total_reward | 301.537180 | 303.964076 | -97.723931 | 917.674726 | 1789 |
