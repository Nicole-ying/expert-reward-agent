# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.014344 | 0.014344 | 0.717207 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.angle_penalty | -0.021651 | 0.021651 | 0.999947 | -0.021652 | 0.021652 | -7.278752 | -0.000000 | 1003520 |
| component.progress_shaping | 0.039643 | 0.044385 | 0.999856 | 0.039649 | 0.044391 | -0.191365 | 0.201610 | 1003520 |
| component.success_landing_bonus | 28.088030 | 28.088030 | 0.280880 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.total_reward | 5.627178 | 5.660766 | 1.000000 | 5.627178 | 5.660766 | -7.318350 | 20.000000 | 1003520 |
| generated_reward | 5.627178 | 5.660766 | 1.000000 | 5.627178 | 5.660766 | -7.318350 | 20.000000 | 1003520 |
| original_env_reward | -0.393556 | 2.260100 | 1.000000 | -0.393556 | 2.260100 | -100.000000 | 121.530401 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -2.053082 | 2.053082 | -18.300000 | -0.320000 | 6999 |
| angle_penalty | -3.102848 | 3.102848 | -96.952430 | -0.002947 | 6999 |
| progress_shaping | 5.681636 | 5.681636 | 0.906267 | 7.085423 | 6999 |
| success_landing_bonus | 4008.829833 | 4008.829833 | 0.000000 | 91400.000000 | 6999 |
| total_reward | 803.138599 | 805.108826 | -88.208913 | 18283.402162 | 6999 |
