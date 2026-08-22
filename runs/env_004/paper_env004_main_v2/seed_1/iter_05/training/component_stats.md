# Reward Component Training Statistics

- steps_seen: 1000448
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.acceleration_bonus | 0.030370 | 0.030370 | 0.584343 | 0.051973 | 0.051973 | 0.000000 | 0.562485 | 1000448 |
| component.forward_progress | 1.871602 | 1.871602 | 0.994633 | 1.881700 | 1.881700 | 0.000000 | 5.127349 | 1000448 |
| component.torso_angle_penalty | -0.040204 | 0.040204 | 1.000000 | -0.040204 | 0.040204 | -0.148058 | -0.000000 | 1000448 |
| component.torso_angular_velocity_penalty | -0.098420 | 0.098420 | 1.000000 | -0.098420 | 0.098420 | -1.000000 | -0.000000 | 1000448 |
| component.total_reward | 1.763348 | 1.764520 | 1.000000 | 1.763348 | 1.764520 | -0.845190 | 4.952183 | 1000448 |
| generated_reward | 1.763348 | 1.764520 | 1.000000 | 1.763348 | 1.764520 | -0.845190 | 4.952183 | 1000448 |
| original_env_reward | 2.861482 | 2.861576 | 1.000000 | 2.861482 | 2.861576 | -2.261226 | 6.112804 | 1000448 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| acceleration_bonus | 7.617200 | 7.617200 | 0.127241 | 32.039681 | 3988 |
| forward_progress | 469.418016 | 469.418016 | 0.000000 | 2268.635459 | 3988 |
| torso_angle_penalty | -10.083317 | 10.083317 | -55.164178 | -0.347714 | 3988 |
| torso_angular_velocity_penalty | -24.684471 | 24.684471 | -138.338412 | -1.479982 | 3988 |
| total_reward | 442.267428 | 442.311487 | -6.880446 | 2141.106207 | 3988 |
