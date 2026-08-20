# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.002796 | 0.002796 | 0.139816 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.angle_penalty | -0.003955 | 0.003955 | 0.999976 | -0.003955 | 0.003955 | -7.200775 | -0.000000 | 1003520 |
| component.progress_shaping | 0.080702 | 0.085284 | 0.999999 | 0.080702 | 0.085284 | -0.203708 | 0.211783 | 1003520 |
| component.success_landing_bonus | 0.022620 | 0.022620 | 0.000226 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.total_reward | 0.078481 | 0.088042 | 1.000000 | 0.078481 | 0.088042 | -7.239231 | 20.000000 | 1003520 |
| generated_reward | 0.078481 | 0.088042 | 1.000000 | 0.078481 | 0.088042 | -7.239231 | 20.000000 | 1003520 |
| original_env_reward | -1.590183 | 2.350104 | 1.000000 | -1.590183 | 2.350104 | -100.000000 | 121.704751 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.195929 | 0.195929 | -2.520000 | 0.000000 | 14321 |
| angle_penalty | -0.277132 | 0.277132 | -83.598701 | -0.000023 | 14321 |
| progress_shaping | 5.654346 | 5.654351 | -0.035176 | 7.030689 | 14321 |
| success_landing_bonus | 1.585085 | 1.585085 | 0.000000 | 100.000000 | 14321 |
| total_reward | 5.498687 | 5.745607 | -80.493161 | 26.317382 | 14321 |
