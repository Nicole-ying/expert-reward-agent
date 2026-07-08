# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_delta | 0.016110 | 0.017043 | 0.999996 | 0.016110 | 0.017043 | -0.040251 | 0.042507 | 1003520 |
| component.soft_landing_proxy | 0.002002 | 0.002002 | 0.004005 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 1003520 |
| component.stability_penalty | -0.137505 | 0.137505 | 1.000000 | -0.137505 | 0.137505 | -0.712710 | -0.000000 | 1003520 |
| component.total_reward | -0.119392 | 0.123311 | 1.000000 | -0.119392 | 0.123311 | -0.735617 | 0.500205 | 1003520 |
| generated_reward | -0.119392 | 0.123311 | 1.000000 | -0.119392 | 0.123311 | -0.735617 | 0.500205 | 1003520 |
| original_env_reward | -1.587342 | 2.423132 | 1.000000 | -1.587342 | 2.423132 | -100.000000 | 130.687253 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_delta | 1.131181 | 1.131203 | -0.157786 | 1.412644 | 14292 |
| soft_landing_proxy | 0.140603 | 0.140603 | 0.000000 | 0.500000 | 14292 |
| stability_penalty | -9.654626 | 9.654626 | -33.786716 | -6.385535 | 14292 |
| total_reward | -8.382842 | 8.382842 | -32.925508 | -4.527701 | 14292 |
