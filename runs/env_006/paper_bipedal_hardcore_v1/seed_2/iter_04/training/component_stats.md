# Reward Component Training Statistics

- steps_seen: 1600000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.balance_penalty | -0.008466 | 0.008466 | 0.052942 | -0.159907 | 0.159907 | -11.304699 | -0.000000 | 1600000 |
| component.forward_reward | 0.106102 | 0.110480 | 0.999999 | 0.106102 | 0.110480 | -0.329508 | 0.669005 | 1600000 |
| component.terrain_gate | 0.497176 | 0.497176 | 1.000000 | 0.497176 | 0.497176 | 0.300000 | 0.957499 | 1600000 |
| component.terrain_roughness | 0.215525 | 0.215525 | 1.000000 | 0.215525 | 0.215525 | 0.018215 | 0.353852 | 1600000 |
| component.total_reward | 0.097636 | 0.114660 | 0.999999 | 0.097636 | 0.114660 | -11.409446 | 0.669005 | 1600000 |
| generated_reward | 0.097636 | 0.114660 | 0.999999 | 0.097636 | 0.114660 | -11.409446 | 0.669005 | 1600000 |
| original_env_reward | -0.489954 | 0.697889 | 1.000000 | -0.489954 | 0.697889 | -100.000000 | 0.771935 | 1600000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| balance_penalty | -1.503190 | 1.503190 | -318.110252 | 0.000000 | 9011 |
| forward_reward | 18.833164 | 18.924289 | -8.280722 | 121.434882 | 9011 |
| terrain_gate | 88.249958 | 88.249958 | 16.510046 | 1232.922591 | 9011 |
| terrain_roughness | 38.255048 | 38.255048 | 7.495695 | 424.388881 | 9011 |
| total_reward | 17.329974 | 18.779559 | -319.107692 | 120.905274 | 9011 |
