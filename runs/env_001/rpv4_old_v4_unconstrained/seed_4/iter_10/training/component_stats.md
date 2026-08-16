# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.002122 | 0.002122 | 0.212164 | -0.010000 | 0.010000 | -0.010000 | 0.000000 | 1003520 |
| component.contact_bonus | 0.010426 | 0.010426 | 0.016805 | 0.620446 | 0.620446 | 0.000000 | 0.999851 | 1003520 |
| component.progress_delta | 0.053631 | 0.057082 | 0.999996 | 0.053631 | 0.057082 | -0.087069 | 0.101789 | 1003520 |
| component.stability_penalty | -0.018710 | 0.018710 | 1.000000 | -0.018710 | 0.018710 | -0.147331 | -0.000000 | 1003520 |
| component.total_reward | 0.043225 | 0.052659 | 1.000000 | 0.043225 | 0.052659 | -0.218640 | 0.991469 | 1003520 |
| generated_reward | 0.043225 | 0.052659 | 1.000000 | 0.043225 | 0.052659 | -0.218640 | 0.991469 | 1003520 |
| original_env_reward | -1.513524 | 2.413905 | 1.000000 | -1.513524 | 2.413905 | -100.000000 | 125.092845 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -0.149021 | 0.149021 | -1.110000 | 0.000000 | 14286 |
| contact_bonus | 0.732409 | 0.732409 | 0.000000 | 1.807167 | 14286 |
| progress_delta | 3.767104 | 3.767297 | -1.375665 | 4.043689 | 14286 |
| stability_penalty | -1.314169 | 1.314169 | -5.236301 | -0.631790 | 14286 |
| total_reward | 3.036323 | 3.048209 | -5.550742 | 4.717914 | 14286 |
