# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.004419 | 0.004419 | 0.999870 | -0.004420 | 0.004420 | -5.321140 | -0.000000 | 1001472 |
| component.proximity_reward | 0.004577 | 0.005824 | 0.981643 | 0.004663 | 0.005933 | -0.089761 | 0.110365 | 1001472 |
| component.settlement_bonus | 0.941939 | 0.941939 | 0.666686 | 1.412868 | 1.412868 | 0.000000 | 1.998317 | 1001472 |
| component.speed_penalty_gated | -0.015902 | 0.015902 | 0.988082 | -0.016093 | 0.016093 | -3.532403 | -0.000000 | 1001472 |
| component.total_reward | 0.926195 | 0.954299 | 1.000000 | 0.926195 | 0.954299 | -6.837425 | 1.998316 | 1001472 |
| generated_reward | 0.926195 | 0.954299 | 1.000000 | 0.926195 | 0.954299 | -6.837425 | 1.998316 | 1001472 |
| original_env_reward | 0.072647 | 1.128056 | 1.000000 | 0.072647 | 1.128056 | -100.000000 | 119.216952 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -2.033022 | 2.033022 | -53.885163 | -0.028791 | 2177 |
| proximity_reward | 2.104378 | 2.364859 | -10.176258 | 2.841820 | 2177 |
| settlement_bonus | 433.210089 | 433.210089 | 0.000000 | 1753.507029 | 2177 |
| speed_penalty_gated | -7.312493 | 7.312493 | -60.757814 | -1.145462 | 2177 |
| total_reward | 425.968952 | 434.543674 | -99.559074 | 1750.604972 | 2177 |
