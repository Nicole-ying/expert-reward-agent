# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_proximity | 0.012032 | 0.012948 | 0.999637 | 0.012036 | 0.012953 | -0.125398 | 0.149225 | 1003520 |
| component.landing_approach | 0.007315 | 0.007524 | 0.606056 | 0.012070 | 0.012415 | -0.024809 | 0.079405 | 1003520 |
| component.total_reward | -0.049940 | 0.056237 | 1.000000 | -0.049940 | 0.056237 | -20.000000 | 0.030281 | 1003520 |
| component.upright_penalty | -0.039958 | 0.039958 | 0.999987 | -0.039959 | 0.039959 | -50.929773 | -0.000000 | 1003520 |
| component.velocity_penalty | -0.031445 | 0.031445 | 0.999749 | -0.031452 | 0.031452 | -2.384671 | -0.000000 | 1003520 |
| generated_reward | -0.049940 | 0.056237 | 1.000000 | -0.049940 | 0.056237 | -20.000000 | 0.030281 | 1003520 |
| original_env_reward | -0.040919 | 1.562879 | 1.000000 | -0.040919 | 1.562879 | -100.000000 | 135.651985 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_proximity | 4.628690 | 4.629135 | -0.579562 | 5.678723 | 2606 |
| landing_approach | 2.814413 | 2.814413 | 2.677209 | 2.844836 | 2606 |
| total_reward | -19.231602 | 20.307674 | -396.567799 | 5.578034 | 2606 |
| upright_penalty | -15.383406 | 15.383406 | -362.308894 | -0.068025 | 2606 |
| velocity_penalty | -12.106026 | 12.106026 | -50.000394 | -1.930665 | 2606 |
