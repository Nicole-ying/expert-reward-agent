# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_velocity_reward | 0.326340 | 0.326340 | 0.909465 | 0.358827 | 0.358827 | 0.000000 | 0.891681 | 1015808 |
| component.total_reward | 0.296424 | 0.342788 | 0.948029 | 0.312674 | 0.361579 | -20.000000 | 0.891681 | 1015808 |
| component.upright_stability_penalty | -0.029974 | 0.029974 | 0.267594 | -0.112013 | 0.112013 | -27.591840 | -0.000000 | 1015808 |
| component.vertical_oscillation_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1015808 |
| generated_reward | 0.296424 | 0.342788 | 0.948029 | 0.312674 | 0.361579 | -20.000000 | 0.891681 | 1015808 |
| original_env_reward | 0.020177 | 0.171196 | 1.000000 | 0.020177 | 0.171196 | -100.000000 | 0.701468 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_velocity_reward | 346.686145 | 346.686145 | 0.000000 | 879.971651 | 945 |
| total_reward | 314.537667 | 354.405406 | -667.687219 | 878.457923 | 945 |
| upright_stability_penalty | -32.210244 | 32.210244 | -671.139867 | -0.230975 | 945 |
| vertical_oscillation_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 945 |
