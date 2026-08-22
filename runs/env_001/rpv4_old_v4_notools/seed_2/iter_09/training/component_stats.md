# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.005576 | 0.005576 | 1.000000 | -0.005576 | 0.005576 | -1.350585 | -0.000000 | 1003520 |
| component.progress | 0.026417 | 0.031245 | 0.999993 | 0.026417 | 0.031245 | -0.245051 | 0.290162 | 1003520 |
| component.stable_contact_bonus | 0.182011 | 0.182011 | 0.364023 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 1003520 |
| component.stable_landing_bonus | 0.473752 | 0.473752 | 0.047375 | 10.000000 | 10.000000 | 0.000000 | 10.000000 | 1003520 |
| component.total_reward | 0.671844 | 0.678355 | 1.000000 | 0.671844 | 0.678355 | -1.555610 | 10.525748 | 1003520 |
| component.velocity_penalty | -0.004760 | 0.004760 | 0.999714 | -0.004761 | 0.004761 | -0.214263 | -0.000000 | 1003520 |
| generated_reward | 0.671844 | 0.678355 | 1.000000 | 0.671844 | 0.678355 | -1.555610 | 10.525748 | 1003520 |
| original_env_reward | -0.075975 | 3.899920 | 1.000000 | -0.075975 | 3.899920 | -100.000000 | 121.242066 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -1.988145 | 1.988145 | -27.266180 | -0.048967 | 2812 |
| progress | 9.412745 | 9.413229 | -0.480074 | 11.311977 | 2812 |
| stable_contact_bonus | 64.791074 | 64.791074 | 0.000000 | 390.500000 | 2812 |
| stable_landing_bonus | 168.669986 | 168.669986 | 0.000000 | 910.000000 | 2812 |
| total_reward | 239.188150 | 239.332908 | -12.360499 | 1183.993565 | 2812 |
| velocity_penalty | -1.697510 | 1.697510 | -7.073558 | -0.448407 | 2812 |
