# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_guidance | 0.333367 | 0.333367 | 0.643840 | 0.517780 | 0.517780 | 0.000000 | 0.599738 | 1003520 |
| component.orientation_penalty | -0.003172 | 0.003172 | 1.000000 | -0.003172 | 0.003172 | -1.097468 | -0.000000 | 1003520 |
| component.progress | 0.001316 | 0.001503 | 0.999197 | 0.001317 | 0.001504 | -0.017204 | 0.018354 | 1003520 |
| component.total_reward | 0.330642 | 0.335023 | 1.000000 | 0.330642 | 0.335023 | -1.115428 | 0.599737 | 1003520 |
| component.velocity_penalty | -0.000870 | 0.000870 | 0.997743 | -0.000872 | 0.000872 | -0.093390 | -0.000000 | 1003520 |
| generated_reward | 0.330642 | 0.335023 | 1.000000 | 0.330642 | 0.335023 | -1.115428 | 0.599737 | 1003520 |
| original_env_reward | 0.017367 | 1.262542 | 1.000000 | 0.017367 | 1.262542 | -100.000000 | 125.504281 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_guidance | 139.164693 | 139.164693 | 0.000000 | 513.148558 | 2397 |
| orientation_penalty | -1.326628 | 1.326628 | -20.691233 | -0.029390 | 2397 |
| progress | 0.550124 | 0.550166 | -0.048599 | 0.709711 | 2397 |
| total_reward | 138.024095 | 139.065465 | -19.658476 | 513.284221 | 2397 |
| velocity_penalty | -0.364093 | 0.364093 | -2.303917 | -0.070945 | 2397 |
