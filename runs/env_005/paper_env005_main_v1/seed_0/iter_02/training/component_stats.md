# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_velocity_reward | 0.315796 | 0.574897 | 1.000000 | 0.315796 | 0.574897 | -4.283610 | 5.538984 | 1003520 |
| component.orientation_reward | 0.091945 | 0.475080 | 1.000000 | 0.091945 | 0.475080 | -0.500000 | 0.500000 | 1003520 |
| component.stability_penalty | -0.247189 | 0.247189 | 1.000000 | -0.247189 | 0.247189 | -3.802310 | -0.000023 | 1003520 |
| component.total_reward | 0.160552 | 0.786414 | 1.000000 | 0.160552 | 0.786414 | -4.336356 | 5.939963 | 1003520 |
| generated_reward | 0.160552 | 0.786414 | 1.000000 | 0.160552 | 0.786414 | -4.336356 | 5.939963 | 1003520 |
| original_env_reward | -0.737474 | 0.972233 | 1.000000 | -0.737474 | 0.972233 | -6.737812 | 4.506510 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_velocity_reward | 61.498479 | 65.900709 | -154.756767 | 953.496385 | 5145 |
| orientation_reward | 17.899853 | 81.480130 | -458.816138 | 490.000399 | 5145 |
| stability_penalty | -48.156926 | 48.156926 | -370.767793 | -1.473681 | 5145 |
| total_reward | 31.241407 | 113.085880 | -663.102446 | 1089.628727 | 5145 |
