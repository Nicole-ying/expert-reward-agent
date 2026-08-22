# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.002001 | 0.002001 | 0.902967 | -0.002216 | 0.002216 | -2.465353 | -0.000000 | 1003520 |
| component.contact_landing_reward | 2.420040 | 2.420040 | 0.588311 | 4.113537 | 4.113537 | 0.000000 | 4.999780 | 1003520 |
| component.lateral_pos_penalty | -0.046161 | 0.046161 | 0.999995 | -0.046161 | 0.046161 | -0.617931 | -0.000000 | 1003520 |
| component.progress_gated | 0.108470 | 0.120328 | 0.998935 | 0.108585 | 0.120456 | -0.952815 | 1.040441 | 1003520 |
| component.total_reward | 2.480348 | 2.503409 | 1.000000 | 2.480348 | 2.503409 | -2.287730 | 5.058210 | 1003520 |
| generated_reward | 2.480348 | 2.503409 | 1.000000 | 2.480348 | 2.503409 | -2.287730 | 5.058210 | 1003520 |
| original_env_reward | -0.065650 | 1.637277 | 1.000000 | -0.065650 | 1.637277 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.542457 | 0.542457 | -8.570416 | -0.002328 | 3698 |
| contact_landing_reward | 655.204450 | 655.204450 | 0.000000 | 4360.522732 | 3698 |
| lateral_pos_penalty | -12.494383 | 12.494383 | -350.645176 | -0.000067 | 3698 |
| progress_gated | 29.410231 | 29.414597 | -2.937830 | 41.029238 | 3698 |
| total_reward | 671.577842 | 672.140152 | -71.346038 | 4394.937773 | 3698 |
