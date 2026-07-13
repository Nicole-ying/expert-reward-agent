# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.003167 | 0.003167 | 0.316728 | -0.010000 | 0.010000 | -0.010000 | 0.000000 | 1003520 |
| component.progress_gate_reward | 0.007542 | 0.007542 | 0.901797 | 0.008363 | 0.008363 | 0.000000 | 0.028858 | 1003520 |
| component.terminal_velocity_penalty | -0.000998 | 0.000998 | 0.003711 | -0.268811 | 0.268811 | -0.793894 | 0.000000 | 1003520 |
| component.total_reward | 0.003377 | 0.007289 | 0.924194 | 0.003654 | 0.007886 | -0.795950 | 0.028582 | 1003520 |
| generated_reward | 0.003377 | 0.007289 | 0.924194 | 0.003654 | 0.007886 | -0.795950 | 0.028582 | 1003520 |
| original_env_reward | -1.070895 | 2.667692 | 1.000000 | -1.070895 | 2.667692 | -100.000000 | 146.284010 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -0.310160 | 0.310160 | -7.410000 | 0.000000 | 10240 |
| progress_gate_reward | 0.738551 | 0.738551 | 0.000846 | 3.828697 | 10240 |
| terminal_velocity_penalty | -0.097759 | 0.097759 | -2.314357 | 0.000000 | 10240 |
| total_reward | 0.330632 | 0.451145 | -6.503664 | 2.020165 | 10240 |
