# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.016459 | 0.017840 | 0.999550 | 0.016466 | 0.017848 | -0.142792 | 0.187959 | 1003520 |
| component.fuel_penalty | -0.007068 | 0.007068 | 0.706821 | -0.010000 | 0.010000 | -0.010000 | 0.000000 | 1003520 |
| component.landing_completion | 0.864231 | 0.864231 | 0.470535 | 1.836699 | 1.836699 | 0.000000 | 1.999940 | 1003520 |
| component.stability | -0.009211 | 0.009211 | 1.000000 | -0.009211 | 0.009211 | -11.972752 | -0.000000 | 1003520 |
| component.total_reward | 0.864410 | 0.875502 | 1.000000 | 0.864410 | 0.875502 | -11.980417 | 1.999953 | 1003520 |
| generated_reward | 0.864410 | 0.875502 | 1.000000 | 0.864410 | 0.875502 | -11.980417 | 1.999953 | 1003520 |
| original_env_reward | -0.007587 | 1.523688 | 1.000000 | -0.007587 | 1.523688 | -100.000000 | 120.593197 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 5.816681 | 5.821723 | -3.275385 | 7.104253 | 2835 |
| fuel_penalty | -2.498250 | 2.498250 | -8.280000 | -0.350000 | 2835 |
| landing_completion | 305.376638 | 305.376638 | 0.000000 | 1663.877823 | 2835 |
| stability | -3.257709 | 3.257709 | -43.889461 | -0.260267 | 2835 |
| total_reward | 305.437360 | 306.315237 | -37.939398 | 1659.103907 | 2835 |
