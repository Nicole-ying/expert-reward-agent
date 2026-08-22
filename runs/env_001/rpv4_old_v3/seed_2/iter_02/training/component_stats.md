# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.005598 | 0.005598 | 0.279910 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.angle_penalty | -0.003545 | 0.003545 | 0.999970 | -0.003545 | 0.003545 | -7.509727 | -0.000000 | 1003520 |
| component.progress_shaping | 0.079910 | 0.084519 | 0.999999 | 0.079910 | 0.084519 | -0.204472 | 0.212149 | 1003520 |
| component.success_landing_bonus | 0.208167 | 0.208167 | 0.002082 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.total_reward | 0.112546 | 0.121760 | 1.000000 | 0.112546 | 0.121760 | -7.569999 | 20.000000 | 1003520 |
| generated_reward | 0.112546 | 0.121760 | 1.000000 | 0.112546 | 0.121760 | -7.569999 | 20.000000 | 1003520 |
| original_env_reward | -1.567727 | 2.383580 | 1.000000 | -1.567727 | 2.383580 | -100.000000 | 135.024787 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.394998 | 0.394998 | -2.540000 | 0.000000 | 14222 |
| angle_penalty | -0.250155 | 0.250155 | -99.206217 | -0.000000 | 14222 |
| progress_shaping | 5.637929 | 5.637941 | -0.090502 | 7.047841 | 14222 |
| success_landing_bonus | 14.688511 | 14.688511 | 0.000000 | 100.000000 | 14222 |
| total_reward | 7.940789 | 8.123665 | -94.860867 | 26.983282 | 14222 |
