# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.001301 | 0.001301 | 0.130138 | -0.010000 | 0.010000 | -0.010000 | -0.000000 | 1003520 |
| component.angle_hinge | -0.001224 | 0.001224 | 0.009552 | -0.128101 | 0.128101 | -1.493356 | -0.000000 | 1003520 |
| component.landing_contact_reward | 0.003208 | 0.003208 | 0.031489 | 0.101879 | 0.101879 | 0.000000 | 0.198984 | 1003520 |
| component.progress_shaping | 0.015437 | 0.017108 | 1.000000 | 0.015437 | 0.017108 | -0.083172 | 1.051914 | 1003520 |
| component.total_reward | 0.016120 | 0.020524 | 1.000000 | 0.016120 | 0.020524 | -1.477187 | 1.141638 | 1003520 |
| generated_reward | 0.016120 | 0.020524 | 1.000000 | 0.016120 | 0.020524 | -1.477187 | 1.141638 | 1003520 |
| original_env_reward | -1.595098 | 2.376518 | 1.000000 | -1.595098 | 2.376518 | -100.000000 | 127.439144 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.091196 | 0.091196 | -1.270000 | 0.000000 | 14319 |
| angle_hinge | -0.085759 | 0.085759 | -36.791295 | 0.000000 | 14319 |
| landing_contact_reward | 0.224829 | 0.224829 | 0.000000 | 1.064220 | 14319 |
| progress_shaping | 1.081831 | 1.082420 | -0.555111 | 1.634608 | 14319 |
| total_reward | 1.129706 | 1.293735 | -37.823802 | 2.047750 | 14319 |
