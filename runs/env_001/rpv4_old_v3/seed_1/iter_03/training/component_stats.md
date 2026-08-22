# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_approach | 0.015918 | 0.016863 | 0.999996 | 0.015918 | 0.016863 | -0.040202 | 0.042588 | 1003520 |
| component.landing_success | 0.535615 | 0.535615 | 0.005356 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.soft_landing | -0.289756 | 0.289756 | 1.000000 | -0.289756 | 0.289756 | -2.154562 | -0.000000 | 1003520 |
| component.total_reward | -0.166582 | 0.380827 | 1.000000 | -0.166582 | 0.380827 | -2.173140 | 20.000000 | 1003520 |
| generated_reward | -0.166582 | 0.380827 | 1.000000 | -0.166582 | 0.380827 | -2.173140 | 20.000000 | 1003520 |
| original_env_reward | -1.574017 | 2.446067 | 1.000000 | -1.574017 | 2.446067 | -100.000000 | 130.408687 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_approach | 1.128421 | 1.128433 | -0.086431 | 1.415186 | 14155 |
| landing_success | 37.972448 | 37.972448 | 0.000000 | 45600.000000 | 14155 |
| soft_landing | -20.540867 | 20.540867 | -85.221112 | -12.971752 | 14155 |
| total_reward | -11.808483 | 16.194411 | -84.827454 | 9061.477600 | 14155 |
