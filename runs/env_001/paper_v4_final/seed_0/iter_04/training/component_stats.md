# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.000186 | 0.000186 | 0.998488 | -0.000186 | 0.000186 | -0.155074 | -0.000000 | 1003520 |
| component.angular_velocity_penalty | -0.000146 | 0.000146 | 0.007362 | -0.019802 | 0.019802 | -0.141572 | -0.000000 | 1003520 |
| component.progress_reward | 0.003602 | 0.003884 | 0.999689 | 0.003603 | 0.003885 | -0.032901 | 0.039326 | 1003520 |
| component.soft_landing | 0.031883 | 0.031883 | 0.774419 | 0.041170 | 0.041170 | 0.000000 | 0.049987 | 1003520 |
| component.total_reward | 0.035153 | 0.035600 | 1.000000 | 0.035153 | 0.035600 | -0.193602 | 0.056030 | 1003520 |
| generated_reward | 0.035153 | 0.035600 | 1.000000 | 0.035153 | 0.035600 | -0.193602 | 0.056030 | 1003520 |
| original_env_reward | -0.081761 | 2.340491 | 1.000000 | -0.081761 | 2.340491 | -100.000000 | 123.737864 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.061644 | 0.061644 | -2.068259 | -0.000039 | 3017 |
| angular_velocity_penalty | -0.048487 | 0.048487 | -0.925128 | 0.000000 | 3017 |
| progress_reward | 1.196504 | 1.196608 | -0.157786 | 1.421014 | 3017 |
| soft_landing | 10.589995 | 10.589995 | 0.000000 | 44.469214 | 3017 |
| total_reward | 11.676368 | 11.684389 | -1.741502 | 45.854779 | 3017 |
