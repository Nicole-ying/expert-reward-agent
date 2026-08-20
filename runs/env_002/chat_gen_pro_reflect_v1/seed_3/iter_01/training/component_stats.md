# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.alternating_bonus | 0.063571 | 0.063571 | 0.635715 | 0.100000 | 0.100000 | 0.000000 | 0.100000 | 1015808 |
| component.balance_penalty | -0.008104 | 0.008104 | 1.000000 | -0.008104 | 0.008104 | -3.795216 | -0.000000 | 1015808 |
| component.both_off_ground_penalty | -0.023830 | 0.023830 | 0.119151 | -0.200000 | 0.200000 | -0.200000 | 0.000000 | 1015808 |
| component.forward_progress_reward | 0.003881 | 0.003881 | 0.999882 | 0.003882 | 0.003882 | 0.000000 | 0.252032 | 1015808 |
| component.total_reward | 0.035518 | 0.090583 | 1.000000 | 0.035518 | 0.090583 | -3.995114 | 0.332320 | 1015808 |
| generated_reward | 0.035518 | 0.090583 | 1.000000 | 0.035518 | 0.090583 | -3.995114 | 0.332320 | 1015808 |
| original_env_reward | -0.077596 | 0.119894 | 1.000000 | -0.077596 | 0.119894 | -100.000000 | 0.622706 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| alternating_bonus | 70.630320 | 70.630320 | 0.800000 | 139.400000 | 907 |
| balance_penalty | -9.046603 | 9.046603 | -136.680824 | -0.326969 | 907 |
| both_off_ground_penalty | -26.571996 | 26.571996 | -83.000000 | -2.400000 | 907 |
| forward_progress_reward | 4.293403 | 4.293403 | 0.016453 | 12.129840 | 907 |
| total_reward | 39.305124 | 49.479250 | -129.989932 | 120.973835 | 907 |
