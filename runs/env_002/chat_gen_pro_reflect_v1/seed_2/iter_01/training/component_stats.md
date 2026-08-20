# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.balance_angle_penalty | -0.001297 | 0.001297 | 0.030018 | -0.043197 | 0.043197 | -2.968704 | -0.000000 | 1015808 |
| component.balance_angular_vel_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1015808 |
| component.forward_progress_reward | 0.077690 | 0.077690 | 0.999986 | 0.077691 | 0.077691 | 0.000000 | 0.510865 | 1015808 |
| component.gate_factor | 0.805243 | 0.805243 | 0.991498 | 0.812148 | 0.812148 | 0.000000 | 1.000000 | 1015808 |
| component.gated_forward_reward | 0.067670 | 0.067670 | 0.991484 | 0.068251 | 0.068251 | 0.000000 | 0.492671 | 1015808 |
| component.total_reward | 0.066373 | 0.068931 | 0.999986 | 0.066374 | 0.068932 | -2.968704 | 0.492671 | 1015808 |
| generated_reward | 0.066373 | 0.068931 | 0.999986 | 0.066374 | 0.068932 | -2.968704 | 0.492671 | 1015808 |
| original_env_reward | 0.067850 | 0.171472 | 1.000000 | 0.067850 | 0.171472 | -100.000000 | 0.680950 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| balance_angle_penalty | -1.496823 | 1.496823 | -61.961846 | 0.000000 | 880 |
| balance_angular_vel_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 880 |
| forward_progress_reward | 88.798458 | 88.798458 | 0.016937 | 245.756402 | 880 |
| gate_factor | 925.452992 | 925.452992 | 6.827282 | 1382.903896 | 880 |
| gated_forward_reward | 77.296807 | 77.296807 | 0.000965 | 228.979711 | 880 |
| total_reward | 75.799984 | 78.385298 | -61.777827 | 228.979711 | 880 |
