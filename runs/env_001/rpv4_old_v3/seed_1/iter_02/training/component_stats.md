# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crash_avoidance | -0.002142 | 0.002142 | 0.000214 | -10.000000 | 10.000000 | -10.000000 | 0.000000 | 1003520 |
| component.goal_approach | 0.016060 | 0.016992 | 0.999995 | 0.016060 | 0.016992 | -0.041140 | 0.042508 | 1003520 |
| component.soft_landing | -0.282052 | 0.282052 | 1.000000 | -0.282052 | 0.282052 | -2.239116 | -0.000000 | 1003520 |
| component.total_reward | -0.268135 | 0.268135 | 1.000000 | -0.268135 | 0.268135 | -12.262240 | 0.001217 | 1003520 |
| generated_reward | -0.268135 | 0.268135 | 1.000000 | -0.268135 | 0.268135 | -12.262240 | 0.001217 | 1003520 |
| original_env_reward | -1.565814 | 2.380030 | 1.000000 | -1.565814 | 2.380030 | -100.000000 | 123.430614 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crash_avoidance | -0.150560 | 0.150560 | -120.000000 | 0.000000 | 14280 |
| goal_approach | 1.128476 | 1.128479 | -0.021245 | 1.410900 | 14280 |
| soft_landing | -19.818469 | 19.818469 | -85.221112 | -12.716757 | 14280 |
| total_reward | -18.840553 | 18.840553 | -167.678576 | -11.359982 | 14280 |
