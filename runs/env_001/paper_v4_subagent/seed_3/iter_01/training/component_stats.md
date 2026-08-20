# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_bonus | 0.057448 | 0.057448 | 0.005745 | 10.000000 | 10.000000 | 0.000000 | 10.000000 | 1003520 |
| component.lateral_drift_penalty | -0.033090 | 0.033090 | 0.997199 | -0.033183 | 0.033183 | -0.907025 | -0.000000 | 1003520 |
| component.progress | 0.146557 | 0.156859 | 0.999996 | 0.146558 | 0.156860 | -0.393518 | 0.498672 | 1003520 |
| component.stability_penalty | -0.027644 | 0.027644 | 1.000000 | -0.027644 | 0.027644 | -7.456011 | -0.000000 | 1003520 |
| component.total_reward | 0.143271 | 0.210801 | 1.000000 | 0.143271 | 0.210801 | -8.139390 | 10.034712 | 1003520 |
| generated_reward | 0.143271 | 0.210801 | 1.000000 | 0.143271 | 0.210801 | -8.139390 | 10.034712 | 1003520 |
| original_env_reward | -1.244945 | 2.563050 | 1.000000 | -1.244945 | 2.563050 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 4.493725 | 4.493725 | 0.000000 | 4370.000000 | 12829 |
| lateral_drift_penalty | -2.588168 | 2.588168 | -26.695811 | -0.000620 | 12829 |
| progress | 11.463389 | 11.463389 | 0.589312 | 14.081420 | 12829 |
| stability_penalty | -2.162224 | 2.162224 | -101.056983 | -0.003682 | 12829 |
| total_reward | 11.206722 | 12.213580 | -91.330881 | 4351.953005 | 12829 |
