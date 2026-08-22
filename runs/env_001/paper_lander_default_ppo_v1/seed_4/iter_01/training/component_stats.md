# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 0.001489 | 0.001489 | 0.005884 | 0.253009 | 0.253009 | 0.000000 | 0.481989 | 1001472 |
| component.orientation_penalty | -0.006182 | 0.006182 | 1.000000 | -0.006182 | 0.006182 | -4.671264 | -0.000000 | 1001472 |
| component.proximity_reward | -0.970673 | 0.970673 | 1.000000 | -0.970673 | 0.970673 | -1.660356 | -0.000960 | 1001472 |
| component.speed_penalty_gated | -0.166140 | 0.166140 | 0.999999 | -0.166140 | 0.166140 | -1.771932 | -0.000000 | 1001472 |
| component.total_reward | -1.141506 | 1.141510 | 1.000000 | -1.141506 | 1.141510 | -5.780586 | 0.123174 | 1001472 |
| generated_reward | -1.141506 | 1.141510 | 1.000000 | -1.141506 | 1.141510 | -5.780586 | 0.123174 | 1001472 |
| original_env_reward | -1.362595 | 2.718934 | 1.000000 | -1.362595 | 2.718934 | -100.000000 | 143.499993 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 0.106476 | 0.106476 | 0.000000 | 99.406075 | 14003 |
| orientation_penalty | -0.442156 | 0.442156 | -54.889540 | -0.000001 | 14003 |
| proximity_reward | -69.416996 | 69.416996 | -409.306138 | -41.994827 | 14003 |
| speed_penalty_gated | -11.881338 | 11.881338 | -39.366525 | -7.305348 | 14003 |
| total_reward | -81.634014 | 81.634014 | -327.154453 | -54.260075 | 14003 |
