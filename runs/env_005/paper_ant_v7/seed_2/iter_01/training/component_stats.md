# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_energy_penalty | -0.042147 | 0.042147 | 1.000000 | -0.042147 | 0.042147 | -0.080000 | -0.001401 | 1001472 |
| component.forward_velocity_reward | 0.744525 | 1.389858 | 1.000000 | 0.744525 | 1.389858 | -8.528478 | 11.235275 | 1001472 |
| component.height_health_penalty | -0.063074 | 0.063074 | 0.386887 | -0.163029 | 0.163029 | -2.278168 | -0.000000 | 1001472 |
| component.lateral_drift_penalty | -0.290986 | 0.290986 | 0.999998 | -0.290987 | 0.290987 | -13.793724 | -0.000000 | 1001472 |
| component.total_reward | -1.162586 | 2.659629 | 1.000000 | -1.162586 | 2.659629 | -14.782544 | 11.097837 | 1001472 |
| component.upright_orientation_penalty | -1.510904 | 1.510904 | 0.999935 | -1.511002 | 1.511002 | -4.000000 | -0.000000 | 1001472 |
| generated_reward | -1.162586 | 2.659629 | 1.000000 | -1.162586 | 2.659629 | -14.782544 | 11.097837 | 1001472 |
| original_env_reward | -0.740758 | 1.054285 | 1.000000 | -0.740758 | 1.054285 | -6.285537 | 5.032777 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_energy_penalty | -3.495631 | 3.495631 | -44.658337 | -0.245955 | 12064 |
| forward_velocity_reward | 61.790527 | 68.945831 | -271.289343 | 1085.057693 | 12064 |
| height_health_penalty | -5.227998 | 5.227998 | -132.163127 | -0.500344 | 12064 |
| lateral_drift_penalty | -24.148362 | 24.148362 | -271.397706 | -0.101092 | 12064 |
| total_reward | -96.265379 | 182.307386 | -4070.728519 | 790.098139 | 12064 |
| upright_orientation_penalty | -125.183916 | 125.183916 | -3734.548926 | -0.000237 | 12064 |
