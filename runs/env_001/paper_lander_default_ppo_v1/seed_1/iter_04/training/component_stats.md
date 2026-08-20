# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_reward | 0.033276 | 0.033276 | 0.902378 | 0.036876 | 0.036876 | 0.000000 | 0.083583 | 1001472 |
| component.soft_landing_bonus | 0.011616 | 0.011616 | 0.007015 | 1.655965 | 1.655965 | 0.000000 | 1.971057 | 1001472 |
| component.stability_penalty | -0.014041 | 0.014041 | 1.000000 | -0.014041 | 0.014041 | -0.072292 | -0.000000 | 1001472 |
| component.total_reward | 0.030851 | 0.034017 | 1.000000 | 0.030851 | 0.034017 | -0.072292 | 1.972304 | 1001472 |
| generated_reward | 0.030851 | 0.034017 | 1.000000 | 0.030851 | 0.034017 | -0.072292 | 1.972304 | 1001472 |
| original_env_reward | -1.401648 | 2.462384 | 1.000000 | -1.401648 | 2.462384 | -100.000000 | 132.122254 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_reward | 2.329903 | 2.329903 | 0.528538 | 3.040283 | 14302 |
| soft_landing_bonus | 0.813394 | 0.813394 | 0.000000 | 5.186014 | 14302 |
| stability_penalty | -0.983160 | 0.983160 | -2.939953 | -0.629455 | 14302 |
| total_reward | 2.160136 | 2.162167 | -1.574324 | 6.184835 | 14302 |
