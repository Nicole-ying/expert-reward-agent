# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 17.423515 | 17.423515 | 0.182205 | 95.626079 | 95.626079 | 0.000000 | 501.000000 | 1003520 |
| component.goal_proximity | 0.015070 | 0.016756 | 0.999991 | 0.015071 | 0.016756 | -0.076646 | 0.082921 | 1003520 |
| component.orientation_penalty | -0.004962 | 0.004962 | 1.000000 | -0.004962 | 0.004962 | -0.335501 | -0.000000 | 1003520 |
| component.total_reward | 0.754590 | 0.918160 | 1.000000 | 0.754590 | 0.918160 | -1.300369 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.094874 | 0.094874 | 0.999077 | -0.094962 | 0.094962 | -1.159899 | -0.000000 | 1003520 |
| generated_reward | 0.754590 | 0.918160 | 1.000000 | 0.754590 | 0.918160 | -1.300369 | 20.000000 | 1003520 |
| original_env_reward | -0.542464 | 3.228244 | 1.000000 | -0.542464 | 3.228244 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 2548.866501 | 2548.866501 | 0.000000 | 52043.000000 | 6839 |
| goal_proximity | 2.210118 | 2.210118 | 0.117863 | 2.824358 | 6839 |
| orientation_penalty | -0.727479 | 0.727479 | -23.465681 | -0.013590 | 6839 |
| total_reward | 110.340458 | 119.471044 | -60.638905 | 2493.746034 | 6839 |
| velocity_penalty | -13.918464 | 13.918464 | -57.069932 | -4.960145 | 6839 |
