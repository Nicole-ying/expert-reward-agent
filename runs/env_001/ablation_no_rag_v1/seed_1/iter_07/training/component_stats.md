# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.158954 | 0.168516 | 0.999996 | 0.158954 | 0.168517 | -0.407151 | 0.417737 | 1003520 |
| component.landing_guidance | 0.005322 | 0.062021 | 0.999963 | 0.005322 | 0.062023 | -0.165826 | 2.000000 | 1003520 |
| component.tilt_penalty | -0.002913 | 0.002913 | 1.000000 | -0.002913 | 0.002913 | -0.376776 | -0.000000 | 1003520 |
| component.total_reward | 0.161362 | 0.173958 | 1.000000 | 0.161362 | 0.173958 | -0.543780 | 2.304654 | 1003520 |
| generated_reward | 0.161362 | 0.173958 | 1.000000 | 0.161362 | 0.173958 | -0.543780 | 2.304654 | 1003520 |
| original_env_reward | -1.430580 | 2.469208 | 1.000000 | -1.430580 | 2.469208 | -100.000000 | 133.191653 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 11.266437 | 11.266437 | 3.296909 | 14.059494 | 14158 |
| landing_guidance | 0.377245 | 1.407087 | -3.098822 | 7.601303 | 14158 |
| tilt_penalty | -0.206492 | 0.206492 | -9.929545 | -0.010716 | 14158 |
| total_reward | 11.437190 | 11.445769 | -7.193704 | 17.702224 | 14158 |
