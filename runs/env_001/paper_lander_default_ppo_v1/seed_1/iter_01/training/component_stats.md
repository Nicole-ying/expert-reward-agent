# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_reward | 0.033320 | 0.033320 | 0.904380 | 0.036843 | 0.036843 | 0.000000 | 0.084855 | 1001472 |
| component.soft_landing_bonus | 0.004871 | 0.004871 | 0.004871 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1001472 |
| component.stability_penalty | -0.136017 | 0.136017 | 1.000000 | -0.136017 | 0.136017 | -0.759639 | -0.000000 | 1001472 |
| component.total_reward | -0.097826 | 0.107508 | 1.000000 | -0.097826 | 0.107508 | -0.759639 | 1.000659 | 1001472 |
| generated_reward | -0.097826 | 0.107508 | 1.000000 | -0.097826 | 0.107508 | -0.759639 | 1.000659 | 1001472 |
| original_env_reward | -1.495976 | 2.357925 | 1.000000 | -1.495976 | 2.357925 | -100.000000 | 132.122254 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_reward | 2.329343 | 2.329343 | 0.297644 | 3.034842 | 14325 |
| soft_landing_bonus | 0.340524 | 0.340524 | 0.000000 | 1.000000 | 14325 |
| stability_penalty | -9.508846 | 9.508846 | -27.461435 | -6.312510 | 14325 |
| total_reward | -6.838979 | 6.838979 | -25.221731 | -2.525010 | 14325 |
