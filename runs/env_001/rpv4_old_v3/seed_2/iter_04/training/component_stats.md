# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.008084 | 0.008084 | 0.404184 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.angle_penalty | -0.004450 | 0.004450 | 0.999972 | -0.004450 | 0.004450 | -7.200775 | -0.000000 | 1003520 |
| component.progress_shaping | 0.079050 | 0.083723 | 0.999994 | 0.079050 | 0.083723 | -0.209484 | 0.212604 | 1003520 |
| component.success_landing_bonus | 0.449019 | 0.449019 | 0.004490 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.total_reward | 0.156585 | 0.167607 | 1.000000 | 0.156585 | 0.167607 | -7.239231 | 20.000000 | 1003520 |
| generated_reward | 0.156585 | 0.167607 | 1.000000 | 0.156585 | 0.167607 | -7.239231 | 20.000000 | 1003520 |
| original_env_reward | -1.486027 | 2.458166 | 1.000000 | -1.486027 | 2.458166 | -100.000000 | 134.240767 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.575018 | 0.575018 | -2.520000 | -0.120000 | 14106 |
| angle_penalty | -0.316566 | 0.316566 | -83.598701 | -0.000232 | 14106 |
| progress_shaping | 5.622934 | 5.622934 | 0.433657 | 6.976346 | 14106 |
| success_landing_bonus | 31.943854 | 31.943854 | 0.000000 | 900.000000 | 14106 |
| total_reward | 11.138987 | 11.416424 | -80.493161 | 173.649102 | 14106 |
