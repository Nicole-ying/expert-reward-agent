# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.156455 | 0.156455 | 0.782277 | -0.200000 | 0.200000 | -0.200000 | -0.000000 | 1003520 |
| component.progress_gated | 0.368101 | 0.368101 | 0.665275 | 0.553306 | 0.553306 | 0.000000 | 1.990967 | 1003520 |
| component.proximity_stability | 5.560330 | 5.560330 | 0.675395 | 8.232712 | 8.232712 | 0.000000 | 14.977671 | 1003520 |
| component.total_reward | 5.771975 | 5.792918 | 0.990421 | 5.827801 | 5.848947 | -0.200000 | 14.977671 | 1003520 |
| generated_reward | 5.771975 | 5.792918 | 0.990421 | 5.827801 | 5.848947 | -0.200000 | 14.977671 | 1003520 |
| original_env_reward | -0.011821 | 2.290366 | 1.000000 | -0.011821 | 2.290366 | -100.000000 | 128.015483 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -60.116916 | 60.116916 | -171.800000 | -5.600000 | 2607 |
| progress_gated | 141.386381 | 141.386381 | 0.000000 | 449.115270 | 2607 |
| proximity_stability | 2135.906358 | 2135.906358 | 0.000000 | 12507.685380 | 2607 |
| total_reward | 2217.175846 | 2218.075544 | -88.325365 | 12587.643894 | 2607 |
