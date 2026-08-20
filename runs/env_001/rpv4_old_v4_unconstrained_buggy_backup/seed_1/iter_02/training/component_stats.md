# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.020116 | 0.020116 | 0.999998 | -0.020116 | 0.020116 | -7.884284 | -0.000000 | 1003520 |
| component.proximity_progress | 0.391799 | 0.419251 | 0.999991 | 0.391802 | 0.419255 | -1.015657 | 1.185095 | 1003520 |
| component.soft_landing_bonus | 0.114539 | 0.114539 | 0.015533 | 7.373742 | 7.373742 | 0.000000 | 19.991261 | 1003520 |
| component.total_reward | 0.486222 | 0.533857 | 1.000000 | 0.486222 | 0.533857 | -8.576806 | 19.942128 | 1003520 |
| generated_reward | 0.486222 | 0.533857 | 1.000000 | 0.486222 | 0.533857 | -8.576806 | 19.942128 | 1003520 |
| original_env_reward | -0.788132 | 2.851410 | 1.000000 | -0.788132 | 2.851410 | -100.000000 | 123.801057 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -1.730471 | 1.730471 | -105.566344 | -0.003108 | 11664 |
| proximity_progress | 33.695364 | 33.695364 | 7.715388 | 42.127120 | 11664 |
| soft_landing_bonus | 9.802985 | 9.802985 | 0.000000 | 1522.735731 | 11664 |
| total_reward | 41.767877 | 41.978837 | -84.736693 | 1558.214683 | 11664 |
