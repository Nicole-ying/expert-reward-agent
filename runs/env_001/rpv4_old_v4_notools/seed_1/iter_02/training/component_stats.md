# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.016914 | 0.016914 | 0.999995 | -0.016914 | 0.016914 | -6.725627 | -0.000000 | 1003520 |
| component.proximity_progress | 0.257178 | 0.276134 | 0.997083 | 0.257931 | 0.276942 | -1.164096 | 1.229590 | 1003520 |
| component.soft_landing_bonus | 1.013300 | 1.013300 | 0.263222 | 3.849594 | 3.849594 | 0.000000 | 4.999878 | 1003520 |
| component.total_reward | 1.253564 | 1.287647 | 1.000000 | 1.253564 | 1.287647 | -6.842399 | 5.013889 | 1003520 |
| generated_reward | 1.253564 | 1.287647 | 1.000000 | 1.253564 | 1.287647 | -6.842399 | 5.013889 | 1003520 |
| original_env_reward | -0.431441 | 2.048925 | 1.000000 | -0.431441 | 2.048925 | -100.000000 | 136.092835 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -2.274314 | 2.274314 | -117.441681 | -0.006548 | 7462 |
| proximity_progress | 34.570263 | 34.570263 | 6.385602 | 42.584338 | 7462 |
| soft_landing_bonus | 135.976384 | 135.976384 | 0.000000 | 4264.291907 | 7462 |
| total_reward | 168.272333 | 168.529687 | -94.820011 | 4304.480697 | 7462 |
