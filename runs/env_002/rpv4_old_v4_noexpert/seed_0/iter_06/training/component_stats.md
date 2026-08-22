# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.037361 | 0.037361 | 1.000000 | -0.037361 | 0.037361 | -0.080000 | -0.000025 | 1015808 |
| component.forward_progress | 0.246946 | 0.254292 | 1.000000 | 0.246946 | 0.254292 | -0.545425 | 0.811379 | 1015808 |
| component.torso_stability | -0.017318 | 0.017318 | 1.000000 | -0.017318 | 0.017318 | -1.745424 | -0.000000 | 1015808 |
| component.total_reward | 0.191654 | 0.216233 | 1.000000 | 0.191654 | 0.216233 | -1.796217 | 0.790786 | 1015808 |
| component.vertical_velocity_penalty | -0.000613 | 0.000613 | 0.999945 | -0.000613 | 0.000613 | -0.017906 | -0.000000 | 1015808 |
| generated_reward | 0.191654 | 0.216233 | 1.000000 | 0.191654 | 0.216233 | -1.796217 | 0.790786 | 1015808 |
| original_env_reward | -0.306699 | 0.529196 | 1.000000 | -0.306699 | 0.529196 | -100.000000 | 0.864898 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -9.169523 | 9.169523 | -67.186548 | -1.307249 | 4119 |
| forward_progress | 60.439237 | 61.167934 | -42.621717 | 597.098713 | 4119 |
| torso_stability | -4.247510 | 4.247510 | -48.733254 | -0.138789 | 4119 |
| total_reward | 46.871351 | 50.222611 | -138.996330 | 519.294236 | 4119 |
| vertical_velocity_penalty | -0.150852 | 0.150852 | -0.980793 | -0.044206 | 4119 |
