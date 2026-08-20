# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.000102 | 0.000102 | 0.999789 | -0.000102 | 0.000102 | -0.112302 | -0.000000 | 1003520 |
| component.progress_reward | 0.016169 | 0.017092 | 0.999997 | 0.016169 | 0.017092 | -0.040712 | 0.042304 | 1003520 |
| component.soft_landing | 0.000619 | 0.000619 | 0.005659 | 0.109459 | 0.109459 | 0.000000 | 0.182897 | 1003520 |
| component.total_reward | 0.016686 | 0.017588 | 1.000000 | 0.016686 | 0.017588 | -0.108717 | 0.181235 | 1003520 |
| generated_reward | 0.016686 | 0.017588 | 1.000000 | 0.016686 | 0.017588 | -0.108717 | 0.181235 | 1003520 |
| original_env_reward | -1.595107 | 2.449758 | 1.000000 | -1.595107 | 2.449758 | -100.000000 | 127.989893 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.007148 | 0.007148 | -1.702644 | -0.000010 | 14310 |
| progress_reward | 1.133698 | 1.133720 | -0.157786 | 1.413998 | 14310 |
| soft_landing | 0.043440 | 0.043440 | 0.000000 | 0.182897 | 14310 |
| total_reward | 1.169989 | 1.171065 | -1.071978 | 1.561259 | 14310 |
