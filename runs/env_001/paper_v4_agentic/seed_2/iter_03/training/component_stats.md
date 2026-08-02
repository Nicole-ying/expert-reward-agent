# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_bonus | 0.043606 | 0.043606 | 0.001914 | 22.779415 | 22.779415 | 0.000000 | 73.575256 | 1003520 |
| component.orientation_penalty | -0.078108 | 0.078108 | 1.000000 | -0.078108 | 0.078108 | -177.450500 | -0.000000 | 1003520 |
| component.proximity_delta | 0.781099 | 0.828951 | 0.999995 | 0.781102 | 0.828955 | -1.944433 | 2.362551 | 1003520 |
| component.total_reward | 0.623969 | 0.783555 | 1.000000 | 0.623969 | 0.783555 | -20.000000 | 20.000000 | 1003520 |
| component.velocity_danger | -0.118123 | 0.118123 | 1.000000 | -0.118123 | 0.118123 | -0.899245 | -0.000000 | 1003520 |
| generated_reward | 0.623969 | 0.783555 | 1.000000 | 0.623969 | 0.783555 | -20.000000 | 20.000000 | 1003520 |
| original_env_reward | -1.394028 | 2.468341 | 1.000000 | -1.394028 | 2.468341 | -100.000000 | 133.796405 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 3.130805 | 3.130805 | 0.000000 | 29975.101025 | 13977 |
| orientation_penalty | -5.607933 | 5.607933 | -1624.999505 | -0.016446 | 13977 |
| proximity_delta | 56.070655 | 56.071714 | -7.400024 | 70.498860 | 13977 |
| total_reward | 44.790116 | 49.697610 | -972.766834 | 12203.173034 | 13977 |
| velocity_danger | -8.480026 | 8.480026 | -14.847596 | -5.289855 | 13977 |
