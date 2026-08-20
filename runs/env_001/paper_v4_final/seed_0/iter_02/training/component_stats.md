# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.000108 | 0.000108 | 0.999769 | -0.000108 | 0.000108 | -0.199578 | -0.000000 | 1003520 |
| component.angular_velocity_penalty | -0.000717 | 0.000717 | 0.010058 | -0.071302 | 0.071302 | -0.157835 | -0.000000 | 1003520 |
| component.progress_reward | 0.016117 | 0.017043 | 0.999995 | 0.016117 | 0.017043 | -0.041864 | 0.042503 | 1003520 |
| component.soft_landing | 0.000662 | 0.000662 | 0.005701 | 0.116062 | 0.116062 | 0.000000 | 0.182896 | 1003520 |
| component.total_reward | 0.015953 | 0.018277 | 1.000000 | 0.015953 | 0.018277 | -0.260572 | 0.181488 | 1003520 |
| generated_reward | 0.015953 | 0.018277 | 1.000000 | 0.015953 | 0.018277 | -0.260572 | 0.181488 | 1003520 |
| original_env_reward | -1.592065 | 2.436124 | 1.000000 | -1.592065 | 2.436124 | -100.000000 | 123.795078 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.007586 | 0.007586 | -2.270971 | -0.000009 | 14334 |
| angular_velocity_penalty | -0.050206 | 0.050206 | -1.145351 | 0.000000 | 14334 |
| progress_reward | 1.128137 | 1.128159 | -0.157786 | 1.412644 | 14334 |
| soft_landing | 0.046323 | 0.046323 | 0.000000 | 0.182896 | 14334 |
| total_reward | 1.116669 | 1.119350 | -2.218083 | 1.562207 | 14334 |
