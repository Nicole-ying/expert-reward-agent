# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.011316 | 0.011316 | 0.676086 | -0.016738 | 0.016738 | -0.020000 | 0.000000 | 1003520 |
| component.orientation_penalty | -0.001869 | 0.001869 | 0.999909 | -0.001870 | 0.001870 | -5.957353 | -0.000000 | 1003520 |
| component.progress_reward | -0.003332 | 0.010191 | 0.999995 | -0.003332 | 0.010191 | -0.149340 | 0.147589 | 1003520 |
| component.total_reward | -0.035333 | 0.035419 | 1.000000 | -0.035333 | 0.035419 | -7.505408 | 0.099425 | 1003520 |
| component.vel_penalty | -0.018816 | 0.018816 | 0.999995 | -0.018816 | 0.018816 | -2.203259 | -0.000000 | 1003520 |
| generated_reward | -0.035333 | 0.035419 | 1.000000 | -0.035333 | 0.035419 | -7.505408 | 0.099425 | 1003520 |
| original_env_reward | -1.743512 | 2.444086 | 1.000000 | -1.743512 | 2.444086 | -100.000000 | 107.178428 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -1.758403 | 1.758403 | -6.605000 | -0.315000 | 6456 |
| orientation_penalty | -0.290576 | 0.290576 | -70.716149 | -0.000893 | 6456 |
| progress_reward | -0.517613 | 0.829761 | -25.216827 | 2.825289 | 6456 |
| total_reward | -5.490934 | 5.490934 | -98.916961 | -3.324365 | 6456 |
| vel_penalty | -2.924341 | 2.924341 | -45.853117 | -0.492824 | 6456 |
