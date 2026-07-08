# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.002503 | 0.002503 | 0.125140 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.progress_delta | 0.048513 | 0.051266 | 0.999999 | 0.048513 | 0.051266 | -0.123641 | 0.127043 | 1003520 |
| component.soft_landing_reward | 0.006010 | 0.006010 | 1.000000 | 0.006010 | 0.006010 | 0.000000 | 0.989118 | 1003520 |
| component.stability_penalty | -0.066513 | 0.066513 | 1.000000 | -0.066513 | 0.066513 | -0.448412 | -0.000000 | 1003520 |
| component.total_reward | -0.014492 | 0.034602 | 1.000000 | -0.014492 | 0.034602 | -0.518470 | 0.990214 | 1003520 |
| generated_reward | -0.014492 | 0.034602 | 1.000000 | -0.014492 | 0.034602 | -0.518470 | 0.990214 | 1003520 |
| original_env_reward | -1.595687 | 2.337036 | 1.000000 | -1.595687 | 2.337036 | -100.000000 | 135.519682 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -0.175001 | 0.175001 | -14.860000 | 0.000000 | 14350 |
| progress_delta | 3.392168 | 3.392168 | 0.975875 | 4.185120 | 14350 |
| soft_landing_reward | 0.420302 | 0.420302 | 0.000004 | 323.436438 | 14350 |
| stability_penalty | -4.650812 | 4.650812 | -15.241590 | -3.948816 | 14350 |
| total_reward | -1.013342 | 1.314497 | -10.142518 | 296.316320 | 14350 |
