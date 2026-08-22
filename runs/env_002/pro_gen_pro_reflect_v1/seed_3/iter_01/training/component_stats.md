# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.025080 | 0.025080 | 1.000000 | -0.025080 | 0.025080 | -0.050000 | -0.000029 | 1015808 |
| component.angular_velocity_penalty | -0.001151 | 0.001151 | 0.999938 | -0.001151 | 0.001151 | -0.051004 | -0.000000 | 1015808 |
| component.forward_progress | 0.159766 | 0.170680 | 1.000000 | 0.159766 | 0.170680 | -0.465398 | 0.678990 | 1015808 |
| component.tilt_penalty | -0.087706 | 0.087706 | 0.999995 | -0.087706 | 0.087706 | -56.899812 | -0.000000 | 1015808 |
| component.total_reward | 0.046711 | 0.208565 | 1.000000 | 0.046711 | 0.208565 | -20.000000 | 0.661297 | 1015808 |
| generated_reward | 0.046711 | 0.208565 | 1.000000 | 0.046711 | 0.208565 | -20.000000 | 0.661297 | 1015808 |
| original_env_reward | -0.401802 | 0.556860 | 1.000000 | -0.401802 | 0.556860 | -100.000000 | 0.603795 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -5.371607 | 5.371607 | -43.704449 | -0.832502 | 4720 |
| angular_velocity_penalty | -0.247089 | 0.247089 | -2.920050 | -0.042938 | 4720 |
| forward_progress | 33.997494 | 35.145508 | -25.874816 | 505.082026 | 4720 |
| tilt_penalty | -18.869755 | 18.869755 | -1304.066429 | -2.322608 | 4720 |
| total_reward | 9.698733 | 38.766867 | -1356.362648 | 466.557995 | 4720 |
