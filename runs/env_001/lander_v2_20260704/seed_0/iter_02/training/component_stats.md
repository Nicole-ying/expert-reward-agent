# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_reward | 0.071087 | 0.075027 | 0.999999 | 0.071087 | 0.075027 | -0.190514 | 0.214384 | 1003520 |
| component.soft_landing_proxy | 0.004787 | 0.004787 | 0.004787 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1003520 |
| component.stability_penalty | -0.013625 | 0.013625 | 1.000000 | -0.013625 | 0.013625 | -0.118084 | -0.000000 | 1003520 |
| component.total_reward | 0.062249 | 0.068421 | 1.000000 | 0.062249 | 0.068421 | -0.240276 | 1.003882 | 1003520 |
| generated_reward | 0.062249 | 0.068421 | 1.000000 | 0.062249 | 0.068421 | -0.240276 | 1.003882 | 1003520 |
| original_env_reward | -0.965005 | 2.763705 | 1.000000 | -0.965005 | 2.763705 | -100.000000 | 144.193966 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_reward | 6.155527 | 6.155663 | -0.788928 | 7.063221 | 11587 |
| soft_landing_proxy | 0.414603 | 0.414603 | 0.000000 | 1.000000 | 11587 |
| stability_penalty | -1.179772 | 1.179772 | -3.867625 | -0.661326 | 11587 |
| total_reward | 5.390358 | 5.391501 | -2.769929 | 7.165151 | 11587 |
