# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.029255 | 0.029255 | 0.999998 | -0.029255 | 0.029255 | -7.994827 | -0.000000 | 1003520 |
| component.proximity_progress | 0.462363 | 0.490272 | 0.999996 | 0.462365 | 0.490274 | -1.174801 | 1.262369 | 1003520 |
| component.soft_landing_bonus | 0.829680 | 0.829680 | 0.004148 | 200.000000 | 200.000000 | 0.000000 | 200.000000 | 1003520 |
| component.speed_penalty | -0.024655 | 0.024655 | 0.997251 | -0.024723 | 0.024723 | -0.126977 | -0.000000 | 1003520 |
| component.total_reward | 0.492403 | 0.569873 | 1.000000 | 0.492403 | 0.569873 | -8.198382 | 20.000000 | 1003520 |
| generated_reward | 0.492403 | 0.569873 | 1.000000 | 0.492403 | 0.569873 | -8.198382 | 20.000000 | 1003520 |
| original_env_reward | -1.305012 | 2.607629 | 1.000000 | -1.305012 | 2.607629 | -100.000000 | 137.599466 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -2.134160 | 2.134160 | -98.054084 | -0.002736 | 13756 |
| proximity_progress | 33.727182 | 33.727812 | -4.331300 | 42.292972 | 13756 |
| soft_landing_bonus | 60.526316 | 60.526316 | 0.000000 | 400.000000 | 13756 |
| speed_penalty | -1.798480 | 1.798480 | -3.537519 | -1.028806 | 13756 |
| total_reward | 35.918723 | 36.091372 | -75.376635 | 60.791498 | 13756 |
