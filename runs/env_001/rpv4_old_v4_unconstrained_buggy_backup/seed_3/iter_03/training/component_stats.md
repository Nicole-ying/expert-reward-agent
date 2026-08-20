# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 31.513174 | 31.513174 | 0.315132 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.goal_proximity | 0.007698 | 0.008246 | 0.999560 | 0.007702 | 0.008250 | -0.022208 | 0.031326 | 1003520 |
| component.orientation_penalty | -0.005509 | 0.005509 | 1.000000 | -0.005509 | 0.005509 | -0.326467 | -0.000000 | 1003520 |
| component.total_reward | 6.251365 | 6.353950 | 1.000000 | 6.251365 | 6.353950 | -0.862468 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.055839 | 0.055839 | 0.998966 | -0.055897 | 0.055897 | -0.694674 | -0.000000 | 1003520 |
| generated_reward | 6.251365 | 6.353950 | 1.000000 | 6.251365 | 6.353950 | -0.862468 | 20.000000 | 1003520 |
| original_env_reward | -0.339881 | 2.102859 | 1.000000 | -0.339881 | 2.102859 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 4727.460674 | 4727.460674 | 0.000000 | 89100.000000 | 6675 |
| goal_proximity | 1.156775 | 1.156775 | 0.405867 | 1.213811 | 6675 |
| orientation_penalty | -0.827754 | 0.827754 | -22.346290 | -0.009788 | 6675 |
| total_reward | 937.786244 | 943.071187 | -21.953224 | 17815.184364 | 6675 |
| velocity_penalty | -8.392800 | 8.392800 | -25.438926 | -2.903725 | 6675 |
