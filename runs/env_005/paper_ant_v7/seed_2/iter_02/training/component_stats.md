# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_energy_penalty | -0.049167 | 0.049167 | 1.000000 | -0.049167 | 0.049167 | -0.080000 | -0.002012 | 1001472 |
| component.forward_velocity_reward | 0.463995 | 0.689649 | 0.522123 | 0.888669 | 1.320854 | -8.214886 | 10.692883 | 1001472 |
| component.height_health_penalty | -0.068500 | 0.068500 | 0.452144 | -0.151500 | 0.151500 | -1.732231 | -0.000000 | 1001472 |
| component.lateral_drift_penalty | -0.169572 | 0.169572 | 0.999940 | -0.169582 | 0.169582 | -12.566397 | -0.000000 | 1001472 |
| component.total_reward | 0.176756 | 0.764986 | 1.000000 | 0.176756 | 0.764986 | -12.612974 | 10.621885 | 1001472 |
| generated_reward | 0.176756 | 0.764986 | 1.000000 | 0.176756 | 0.764986 | -12.612974 | 10.621885 | 1001472 |
| original_env_reward | -1.215829 | 1.326633 | 1.000000 | -1.215829 | 1.326633 | -6.236661 | 4.334428 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_energy_penalty | -8.724553 | 8.724553 | -67.214556 | -0.189364 | 5643 |
| forward_velocity_reward | 82.318238 | 88.503312 | -218.207153 | 1346.221794 | 5643 |
| height_health_penalty | -12.156750 | 12.156750 | -143.556859 | 0.000000 | 5643 |
| lateral_drift_penalty | -30.085498 | 30.085498 | -332.944564 | -0.074360 | 5643 |
| total_reward | 31.351437 | 76.244143 | -500.488421 | 1057.676507 | 5643 |
