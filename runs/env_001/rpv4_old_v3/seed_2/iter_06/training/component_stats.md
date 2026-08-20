# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.002269 | 0.002269 | 0.113468 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.angle_penalty | -0.001847 | 0.001847 | 0.999950 | -0.001847 | 0.001847 | -7.200775 | -0.000000 | 1003520 |
| component.progress_shaping | 0.080677 | 0.085295 | 0.999997 | 0.080677 | 0.085295 | -0.203348 | 0.211768 | 1003520 |
| component.success_landing_bonus | 0.004982 | 0.004982 | 0.000025 | 200.000000 | 200.000000 | 0.000000 | 200.000000 | 1003520 |
| component.total_reward | 0.077059 | 0.084766 | 1.000000 | 0.077059 | 0.084766 | -7.239231 | 20.000000 | 1003520 |
| generated_reward | 0.077059 | 0.084766 | 1.000000 | 0.077059 | 0.084766 | -7.239231 | 20.000000 | 1003520 |
| original_env_reward | -1.584312 | 2.287885 | 1.000000 | -1.584312 | 2.287885 | -100.000000 | 123.957331 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.158814 | 0.158814 | -14.920000 | 0.000000 | 14338 |
| angle_penalty | -0.129291 | 0.129291 | -83.598701 | -0.000000 | 14338 |
| progress_shaping | 5.646303 | 5.646303 | 0.878474 | 6.941339 | 14338 |
| success_landing_bonus | 0.348724 | 0.348724 | 0.000000 | 200.000000 | 14338 |
| total_reward | 5.393146 | 5.510806 | -80.493161 | 26.301121 | 14338 |
