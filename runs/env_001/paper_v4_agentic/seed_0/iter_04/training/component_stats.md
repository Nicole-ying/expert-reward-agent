# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_safety_penalty | 0.015900 | 0.015900 | 1.000000 | 0.015900 | 0.015900 | 0.000000 | 0.374797 | 1003520 |
| component.precise_landing_bonus | 0.066338 | 0.066338 | 0.006026 | 11.009004 | 11.009004 | 0.000000 | 18.288988 | 1003520 |
| component.progress_reward | 0.015547 | 0.016503 | 0.999996 | 0.015548 | 0.016503 | -0.040873 | 0.042377 | 1003520 |
| component.total_reward | 0.065986 | 0.078622 | 0.999999 | 0.065986 | 0.078622 | -0.354935 | 18.285521 | 1003520 |
| component.x_boundary_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1003520 |
| generated_reward | 0.065986 | 0.078622 | 0.999999 | 0.065986 | 0.078622 | -0.354935 | 18.285521 | 1003520 |
| original_env_reward | -1.467751 | 2.489650 | 1.000000 | -1.467751 | 2.489650 | -100.000000 | 110.746283 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_safety_penalty | 1.148921 | 1.148921 | 0.575480 | 6.098182 | 13887 |
| precise_landing_bonus | 4.793796 | 4.793796 | 0.000000 | 18.288988 | 13887 |
| progress_reward | 1.123436 | 1.123459 | -0.157786 | 1.412644 | 13887 |
| total_reward | 4.768311 | 4.925285 | -4.701220 | 18.497070 | 13887 |
| x_boundary_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 13887 |
