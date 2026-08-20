# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.001137 | 0.001137 | 0.008993 | -0.126481 | 0.126481 | -1.717435 | -0.000000 | 1003520 |
| component.distance_delta | 0.030743 | 0.034662 | 0.998154 | 0.030800 | 0.034726 | -0.381806 | 0.396812 | 1003520 |
| component.engine_penalty | -0.006853 | 0.006853 | 0.685317 | -0.010000 | 0.010000 | -0.010000 | -0.000000 | 1003520 |
| component.soft_landing_progress | 0.728490 | 0.728490 | 1.000000 | 0.728490 | 0.728490 | 0.000113 | 1.993742 | 1003520 |
| component.total_reward | 0.751242 | 0.753669 | 1.000000 | 0.751242 | 0.753669 | -1.804187 | 1.992364 | 1003520 |
| generated_reward | 0.751242 | 0.753669 | 1.000000 | 0.751242 | 0.753669 | -1.804187 | 1.992364 | 1003520 |
| original_env_reward | -0.114525 | 1.856453 | 1.000000 | -0.114525 | 1.856453 | -100.000000 | 126.130579 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.405071 | 0.405071 | -38.746215 | 0.000000 | 2818 |
| distance_delta | 10.929989 | 11.072833 | -85.643002 | 14.185309 | 2818 |
| engine_penalty | -2.434904 | 2.434904 | -9.170000 | -0.240000 | 2818 |
| soft_landing_progress | 258.286800 | 258.286800 | 0.229528 | 1620.793698 | 2818 |
| total_reward | 266.376814 | 266.665319 | -58.298694 | 1628.524318 | 2818 |
