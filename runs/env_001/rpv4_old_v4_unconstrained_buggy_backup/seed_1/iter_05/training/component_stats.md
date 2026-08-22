# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.030121 | 0.030121 | 0.999998 | -0.030121 | 0.030121 | -11.059217 | -0.000000 | 1003520 |
| component.proximity_progress | 0.806460 | 0.852465 | 0.999999 | 0.806461 | 0.852466 | -1.988294 | 2.103642 | 1003520 |
| component.soft_landing_bonus | 0.015142 | 0.015142 | 0.016149 | 0.937660 | 0.937660 | 0.000000 | 2.499995 | 1003520 |
| component.total_reward | 0.791481 | 0.877583 | 1.000000 | 0.791481 | 0.877583 | -11.098515 | 2.509410 | 1003520 |
| generated_reward | 0.791481 | 0.877583 | 1.000000 | 0.791481 | 0.877583 | -11.098515 | 2.509410 | 1003520 |
| original_env_reward | -1.601692 | 2.418992 | 1.000000 | -1.601692 | 2.418992 | -100.000000 | 135.814025 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -2.110394 | 2.110394 | -135.931254 | -0.002347 | 14323 |
| proximity_progress | 56.498550 | 56.498550 | 8.510270 | 70.693224 | 14323 |
| soft_landing_bonus | 1.060931 | 1.060931 | 0.000000 | 3.255075 | 14323 |
| total_reward | 55.449087 | 55.570486 | -86.189379 | 71.760462 | 14323 |
