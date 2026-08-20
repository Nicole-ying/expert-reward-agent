# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.000115 | 0.000115 | 0.999780 | -0.000115 | 0.000115 | -0.197460 | -0.000000 | 1003520 |
| component.angular_velocity_penalty | -0.000657 | 0.000657 | 0.009705 | -0.067719 | 0.067719 | -0.156748 | -0.000000 | 1003520 |
| component.progress_reward | 0.015740 | 0.016661 | 0.999997 | 0.015740 | 0.016661 | -0.039961 | 0.042086 | 1003520 |
| component.soft_landing | 0.000674 | 0.000674 | 0.006306 | 0.106898 | 0.106898 | 0.000000 | 0.182899 | 1003520 |
| component.total_reward | 0.015642 | 0.017845 | 1.000000 | 0.015642 | 0.017845 | -0.239628 | 0.183842 | 1003520 |
| generated_reward | 0.015642 | 0.017845 | 1.000000 | 0.015642 | 0.017845 | -0.239628 | 0.183842 | 1003520 |
| original_env_reward | -1.320746 | 2.615593 | 1.000000 | -1.320746 | 2.615593 | -100.000000 | 128.155508 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.008254 | 0.008254 | -2.781789 | -0.000004 | 13991 |
| angular_velocity_penalty | -0.047139 | 0.047139 | -1.131977 | 0.000000 | 13991 |
| progress_reward | 1.128725 | 1.128747 | -0.157786 | 1.412644 | 13991 |
| soft_landing | 0.048349 | 0.048349 | 0.000000 | 0.418961 | 13991 |
| total_reward | 1.121681 | 1.124054 | -2.593712 | 1.560572 | 13991 |
