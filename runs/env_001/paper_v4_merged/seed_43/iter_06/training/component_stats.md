# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.001158 | 0.001158 | 0.115814 | -0.010000 | 0.010000 | -0.010000 | -0.000000 | 1003520 |
| component.angle_hinge_penalty | -0.000119 | 0.000119 | 0.018755 | -0.006343 | 0.006343 | -0.089601 | -0.000000 | 1003520 |
| component.landing_contact_reward | 0.003182 | 0.003182 | 0.031953 | 0.099586 | 0.099586 | 0.000000 | 0.198555 | 1003520 |
| component.progress_shaping | 0.015313 | 0.016914 | 1.000000 | 0.015313 | 0.016914 | -0.094629 | 1.049327 | 1003520 |
| component.shaped_progress | 0.012347 | 0.013491 | 1.000000 | 0.012347 | 0.013491 | -0.074300 | 1.049327 | 1003520 |
| component.total_reward | 0.014252 | 0.016200 | 1.000000 | 0.014252 | 0.016200 | -0.144560 | 1.146723 | 1003520 |
| generated_reward | 0.014252 | 0.016200 | 1.000000 | 0.014252 | 0.016200 | -0.144560 | 1.146723 | 1003520 |
| original_env_reward | -1.650330 | 2.409244 | 1.000000 | -1.650330 | 2.409244 | -100.000000 | 133.016045 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.081653 | 0.081653 | -1.140000 | 0.000000 | 14231 |
| angle_hinge_penalty | -0.008389 | 0.008389 | -2.016707 | 0.000000 | 14231 |
| landing_contact_reward | 0.224385 | 0.224385 | 0.000000 | 1.600578 | 14231 |
| progress_shaping | 1.079786 | 1.080304 | -0.714326 | 1.636976 | 14231 |
| shaped_progress | 0.870599 | 0.871117 | -0.714326 | 1.489331 | 14231 |
| total_reward | 1.004942 | 1.016224 | -3.531033 | 1.816727 | 14231 |
