# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.000652 | 0.000652 | 0.065248 | -0.010000 | 0.010000 | -0.010000 | -0.000000 | 1003520 |
| component.angle_hinge_penalty | -0.000132 | 0.000132 | 0.024442 | -0.005410 | 0.005410 | -0.104991 | -0.000000 | 1003520 |
| component.landing_contact_reward | 0.003101 | 0.003101 | 0.032385 | 0.095755 | 0.095755 | 0.000000 | 0.199146 | 1003520 |
| component.progress_shaping | 0.008262 | 0.008309 | 1.000000 | 0.008262 | 0.008309 | -0.002085 | 0.021103 | 1003520 |
| component.shaped_progress | 0.007111 | 0.007150 | 1.000000 | 0.007111 | 0.007150 | -0.001964 | 0.017429 | 1003520 |
| component.total_reward | 0.009428 | 0.010256 | 1.000000 | 0.009428 | 0.010256 | -0.111993 | 0.200862 | 1003520 |
| generated_reward | 0.009428 | 0.010256 | 1.000000 | 0.009428 | 0.010256 | -0.111993 | 0.200862 | 1003520 |
| original_env_reward | -1.726053 | 2.371365 | 1.000000 | -1.726053 | 2.371365 | -100.000000 | 135.667343 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.045647 | 0.045647 | -1.140000 | 0.000000 | 14343 |
| angle_hinge_penalty | -0.009251 | 0.009251 | -2.861702 | 0.000000 | 14343 |
| landing_contact_reward | 0.216965 | 0.216965 | 0.000000 | 2.361724 | 14343 |
| progress_shaping | 0.577969 | 0.577969 | 0.152064 | 0.834469 | 14343 |
| shaped_progress | 0.497472 | 0.497472 | 0.152064 | 0.671707 | 14343 |
| total_reward | 0.659539 | 0.673037 | -3.309511 | 1.462961 | 14343 |
