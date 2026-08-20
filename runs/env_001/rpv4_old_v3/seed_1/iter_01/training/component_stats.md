# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crash_avoidance | -0.000618 | 0.000618 | 0.000062 | -10.000000 | 10.000000 | -10.000000 | 0.000000 | 1003520 |
| component.goal_approach | 0.000780 | 0.003344 | 1.000000 | 0.000780 | 0.003344 | -0.042968 | 0.049272 | 1003520 |
| component.soft_landing | -0.009177 | 0.009177 | 0.008990 | -1.020765 | 1.020765 | -21.256562 | -0.000000 | 1003520 |
| component.total_reward | -0.009013 | 0.013020 | 1.000000 | -0.009013 | 0.013020 | -20.000000 | 0.048414 | 1003520 |
| generated_reward | -0.009013 | 0.013020 | 1.000000 | -0.009013 | 0.013020 | -20.000000 | 0.048414 | 1003520 |
| original_env_reward | -2.950116 | 3.361804 | 1.000000 | -2.950116 | 3.361804 | -100.000000 | 188.419819 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crash_avoidance | -0.081579 | 0.081579 | -90.000000 | 0.000000 | 7600 |
| goal_approach | 0.102883 | 0.277564 | -6.325715 | 1.400226 | 7600 |
| soft_landing | -1.211755 | 1.211755 | -121.684355 | 0.000000 | 7600 |
| total_reward | -1.190244 | 1.468058 | -211.144281 | 0.298724 | 7600 |
