# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.013187 | 0.013187 | 0.659325 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.angle_penalty | -0.014737 | 0.014737 | 0.999990 | -0.014737 | 0.014737 | -6.222020 | -0.000000 | 1003520 |
| component.landing_velocity_penalty | -0.117688 | 0.117688 | 0.134093 | -0.877663 | 0.877663 | -3.511502 | -0.000000 | 1003520 |
| component.progress_shaping | 0.034635 | 0.044610 | 0.998065 | 0.034702 | 0.044697 | -0.160398 | 0.200641 | 1003520 |
| component.total_reward | -0.110977 | 0.153797 | 1.000000 | -0.110977 | 0.153797 | -7.152582 | 0.172321 | 1003520 |
| generated_reward | -0.110977 | 0.153797 | 1.000000 | -0.110977 | 0.153797 | -7.152582 | 0.172321 | 1003520 |
| original_env_reward | -0.859571 | 2.218799 | 1.000000 | -0.859571 | 2.218799 | -100.000000 | 129.118547 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -1.687982 | 1.687982 | -16.300000 | -0.240000 | 7833 |
| angle_penalty | -1.887549 | 1.887549 | -121.302841 | -0.002449 | 7833 |
| landing_velocity_penalty | -15.077563 | 15.077563 | -50.215396 | 0.000000 | 7833 |
| progress_shaping | 4.435312 | 4.490937 | -4.631195 | 7.076994 | 7833 |
| total_reward | -14.217781 | 14.232987 | -155.550843 | 2.453607 | 7833 |
