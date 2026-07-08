# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_delta | 0.016120 | 0.017045 | 0.999998 | 0.016120 | 0.017045 | -0.041569 | 0.042256 | 1003520 |
| component.soft_landing_proxy | 0.003067 | 0.003067 | 0.007923 | 0.387154 | 0.387154 | 0.000000 | 0.492768 | 1003520 |
| component.stability_penalty | -0.137460 | 0.137460 | 1.000000 | -0.137460 | 0.137460 | -0.730042 | -0.000000 | 1003520 |
| component.total_reward | -0.118273 | 0.123963 | 1.000000 | -0.118273 | 0.123963 | -0.748985 | 0.491885 | 1003520 |
| generated_reward | -0.118273 | 0.123963 | 1.000000 | -0.118273 | 0.123963 | -0.748985 | 0.491885 | 1003520 |
| original_env_reward | -1.583943 | 2.398490 | 1.000000 | -1.583943 | 2.398490 | -100.000000 | 131.013083 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_delta | 1.129938 | 1.129960 | -0.157786 | 1.412644 | 14315 |
| soft_landing_proxy | 0.215037 | 0.215037 | 0.000000 | 258.600129 | 14315 |
| stability_penalty | -9.635665 | 9.635665 | -31.170536 | -6.370637 | 14315 |
| total_reward | -8.290690 | 8.322609 | -26.505244 | 228.465073 | 14315 |
