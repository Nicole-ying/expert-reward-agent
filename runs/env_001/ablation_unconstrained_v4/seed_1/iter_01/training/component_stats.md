# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.015024 | 0.016097 | 0.999991 | 0.015024 | 0.016097 | -0.040424 | 0.052429 | 1003520 |
| component.posture_penalty | -0.002471 | 0.002471 | 0.999994 | -0.002471 | 0.002471 | -1.254838 | -0.000000 | 1003520 |
| component.soft_landing_bonus | 0.034521 | 0.034521 | 0.007165 | 4.818085 | 4.818085 | 0.000000 | 9.818912 | 1003520 |
| component.total_reward | 0.047073 | 0.050305 | 1.000000 | 0.047073 | 0.050305 | -1.259918 | 9.814977 | 1003520 |
| generated_reward | 0.047073 | 0.050305 | 1.000000 | 0.047073 | 0.050305 | -1.259918 | 9.814977 | 1003520 |
| original_env_reward | -1.325031 | 2.587836 | 1.000000 | -1.325031 | 2.587836 | -100.000000 | 133.375328 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 1.143461 | 1.143461 | 0.329691 | 1.406786 | 13183 |
| posture_penalty | -0.188082 | 0.188082 | -16.770724 | -0.000331 | 13183 |
| soft_landing_bonus | 2.627780 | 2.627780 | 0.000000 | 9.818912 | 13183 |
| total_reward | 3.583160 | 3.642271 | -16.321824 | 11.188803 | 13183 |
