# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_reward | 0.016226 | 0.017149 | 0.999995 | 0.016226 | 0.017149 | -0.041212 | 0.042493 | 1001472 |
| component.soft_landing_proxy | 0.002695 | 0.002695 | 0.005389 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 1001472 |
| component.stability_penalty | -0.013529 | 0.013529 | 1.000000 | -0.013529 | 0.013529 | -0.059051 | -0.000000 | 1001472 |
| component.total_reward | 0.005391 | 0.010231 | 1.000000 | 0.005391 | 0.010231 | -0.075710 | 0.502619 | 1001472 |
| generated_reward | 0.005391 | 0.010231 | 1.000000 | 0.005391 | 0.010231 | -0.075710 | 0.502619 | 1001472 |
| original_env_reward | -1.578370 | 2.342818 | 1.000000 | -1.578370 | 2.342818 | -100.000000 | 130.550316 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_reward | 1.131378 | 1.131378 | 0.190857 | 1.397777 | 14362 |
| soft_landing_proxy | 0.187892 | 0.187892 | 0.000000 | 3.000000 | 14362 |
| stability_penalty | -0.943336 | 0.943336 | -3.062411 | -0.628384 | 14362 |
| total_reward | 0.375933 | 0.516486 | -2.509514 | 2.236900 | 14362 |
