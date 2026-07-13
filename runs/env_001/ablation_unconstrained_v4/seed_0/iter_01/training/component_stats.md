# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.engine_efficiency | -0.014032 | 0.014032 | 0.140324 | -0.100000 | 0.100000 | -0.100000 | 0.000000 | 1003520 |
| component.orientation_penalty | -0.019809 | 0.019809 | 0.999996 | -0.019809 | 0.019809 | -9.051312 | -0.000000 | 1003520 |
| component.progress_to_goal | -0.971997 | 0.971997 | 1.000000 | -0.971997 | 0.971997 | -1.696205 | -0.000930 | 1003520 |
| component.successful_settle_proxy | 0.034885 | 0.034885 | 0.033807 | 1.031893 | 1.031893 | 0.000000 | 8.238573 | 1003520 |
| component.total_reward | -0.970954 | 1.028248 | 1.000000 | -0.970954 | 1.028248 | -9.216737 | 8.195822 | 1003520 |
| generated_reward | -0.970954 | 1.028248 | 1.000000 | -0.970954 | 1.028248 | -9.216737 | 8.195822 | 1003520 |
| original_env_reward | -1.729039 | 2.458183 | 1.000000 | -1.729039 | 2.458183 | -100.000000 | 135.078524 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| engine_efficiency | -0.982740 | 0.982740 | -71.000000 | 0.000000 | 14328 |
| orientation_penalty | -1.387416 | 1.387416 | -97.606825 | -0.000003 | 14328 |
| progress_to_goal | -68.067207 | 68.067207 | -475.225262 | -40.664154 | 14328 |
| successful_settle_proxy | 2.443327 | 2.443327 | 0.000000 | 1605.609099 | 14328 |
| total_reward | -67.994036 | 68.141165 | -214.730890 | 1054.033046 | 14328 |
