# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.017549 | 0.017549 | 0.999995 | -0.017549 | 0.017549 | -6.095286 | -0.000000 | 1003520 |
| component.proximity_progress | 0.107710 | 0.129282 | 0.999972 | 0.107713 | 0.129286 | -0.987994 | 1.128073 | 1003520 |
| component.soft_landing_bonus | 1.280951 | 1.280951 | 0.062692 | 20.432345 | 20.432345 | 0.000000 | 29.997467 | 1003520 |
| component.total_reward | 1.177130 | 1.214138 | 1.000000 | 1.177130 | 1.214138 | -6.518593 | 20.000000 | 1003520 |
| generated_reward | 1.177130 | 1.214138 | 1.000000 | 1.177130 | 1.214138 | -6.518593 | 20.000000 | 1003520 |
| original_env_reward | -0.144402 | 3.801048 | 1.000000 | -0.144402 | 3.801048 | -100.000000 | 130.086970 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -5.314820 | 5.314820 | -106.074490 | -0.009846 | 3309 |
| proximity_progress | 32.626938 | 32.626938 | 1.045288 | 42.389448 | 3309 |
| soft_landing_bonus | 387.989282 | 387.989282 | 0.000000 | 2776.620559 | 3309 |
| total_reward | 356.495730 | 357.244465 | -58.147021 | 2167.000385 | 3309 |
