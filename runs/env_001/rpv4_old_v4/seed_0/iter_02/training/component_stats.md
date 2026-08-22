# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_proximity_reward | 2.273093 | 2.273093 | 0.999425 | 2.274401 | 2.274401 | 0.000000 | 2.999976 | 1003520 |
| component.orientation_penalty | -0.005457 | 0.005457 | 1.000000 | -0.005457 | 0.005457 | -4.594483 | -0.000000 | 1003520 |
| component.progress_reward | 0.001389 | 0.002416 | 1.000000 | 0.001389 | 0.002416 | -0.040581 | 0.054043 | 1003520 |
| component.total_reward | 2.223000 | 2.232365 | 1.000000 | 2.223000 | 2.232365 | -4.628935 | 2.999879 | 1003520 |
| component.vel_damping | -0.046025 | 0.046025 | 0.818716 | -0.056216 | 0.056216 | -1.905235 | -0.000000 | 1003520 |
| generated_reward | 2.223000 | 2.232365 | 1.000000 | 2.223000 | 2.232365 | -4.628935 | 2.999879 | 1003520 |
| original_env_reward | -0.150611 | 1.913047 | 1.000000 | -0.150611 | 1.913047 | -100.000000 | 111.048727 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_proximity_reward | 1394.028574 | 1394.028574 | 12.638558 | 2801.557084 | 1633 |
| orientation_penalty | -3.352368 | 3.352368 | -66.042620 | -0.053026 | 1633 |
| progress_reward | 0.851468 | 0.924963 | -8.575119 | 1.415539 | 1633 |
| total_reward | 1363.305174 | 1363.366389 | -14.257065 | 2754.492057 | 1633 |
| vel_damping | -28.222500 | 28.222500 | -95.471097 | 0.000000 | 1633 |
