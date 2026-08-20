# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.005063 | 0.005679 | 0.999418 | 0.005066 | 0.005682 | -0.034579 | 0.039975 | 1003520 |
| component.landing_guidance | 0.905328 | 0.924730 | 0.999996 | 0.905332 | 0.924734 | -0.134267 | 2.000000 | 1003520 |
| component.tilt_penalty | -0.007888 | 0.007888 | 1.000000 | -0.007888 | 0.007888 | -0.323025 | -0.000000 | 1003520 |
| component.total_reward | 0.902503 | 0.919974 | 1.000000 | 0.902503 | 0.919974 | -0.353194 | 2.016619 | 1003520 |
| generated_reward | 0.902503 | 0.919974 | 1.000000 | 0.902503 | 0.919974 | -0.353194 | 2.016619 | 1003520 |
| original_env_reward | -0.213112 | 1.641044 | 1.000000 | -0.213112 | 1.641044 | -100.000000 | 141.972585 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 1.062112 | 1.062112 | 0.181439 | 1.419652 | 4780 |
| landing_guidance | 189.922873 | 190.939767 | -3.130995 | 1780.078805 | 4780 |
| tilt_penalty | -1.654108 | 1.654108 | -36.689410 | -0.027347 | 4780 |
| total_reward | 189.330877 | 190.409362 | -11.029006 | 1770.386912 | 4780 |
