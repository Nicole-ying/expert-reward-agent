# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.013523 | 0.013523 | 0.676137 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.progress_reward | 0.009683 | 0.010919 | 0.999954 | 0.009683 | 0.010920 | -0.076385 | 0.080936 | 1003520 |
| component.safe_contact_reward | 1.854207 | 1.854207 | 0.054357 | 34.111860 | 34.111860 | 0.000000 | 99.291740 | 1003520 |
| component.stability_penalty | -0.045403 | 0.045403 | 1.000000 | -0.045403 | 0.045403 | -5.380998 | -0.000000 | 1003520 |
| component.total_reward | 0.818098 | 0.913006 | 0.999999 | 0.818099 | 0.913007 | -5.418930 | 20.000000 | 1003520 |
| generated_reward | 0.818098 | 0.913006 | 0.999999 | 0.818099 | 0.913007 | -5.418930 | 20.000000 | 1003520 |
| original_env_reward | -0.289550 | 3.611408 | 1.000000 | -0.289550 | 3.611408 | -100.000000 | 130.828288 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -3.096553 | 3.096553 | -18.900000 | -0.140000 | 4369 |
| progress_reward | 2.221771 | 2.222006 | -0.448384 | 2.834473 | 4369 |
| safe_contact_reward | 423.683782 | 423.683782 | 0.000000 | 7310.904571 | 4369 |
| stability_penalty | -10.422126 | 10.422126 | -61.795889 | -2.629872 | 4369 |
| total_reward | 186.885855 | 200.531697 | -60.306001 | 2077.828777 | 4369 |
