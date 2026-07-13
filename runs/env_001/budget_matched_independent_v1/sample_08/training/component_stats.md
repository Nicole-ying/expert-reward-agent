# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.019246 | 0.019246 | 0.999971 | -0.019247 | 0.019247 | -11.511029 | -0.000000 | 1003520 |
| component.proximity_reward | -0.121929 | 0.121929 | 1.000000 | -0.121929 | 0.121929 | -0.334483 | -0.000001 | 1003520 |
| component.safe_contact_bonus | 1.178435 | 1.178435 | 0.354458 | 3.324608 | 3.324608 | 0.000000 | 4.999997 | 1003520 |
| component.total_reward | 0.978051 | 1.349231 | 1.000000 | 0.978051 | 1.349231 | -11.760868 | 4.999989 | 1003520 |
| component.velocity_penalty | -0.059208 | 0.059208 | 0.998927 | -0.059272 | 0.059272 | -3.519696 | -0.000000 | 1003520 |
| generated_reward | 0.978051 | 1.349231 | 1.000000 | 0.978051 | 1.349231 | -11.760868 | 4.999989 | 1003520 |
| original_env_reward | -0.350264 | 2.080730 | 1.000000 | -0.350264 | 2.080730 | -100.000000 | 146.587757 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -2.799634 | 2.799634 | -90.363422 | -0.001779 | 6895 |
| proximity_reward | -17.724022 | 17.724022 | -163.737403 | -8.244806 | 6895 |
| safe_contact_bonus | 171.276886 | 171.276886 | 0.000000 | 4459.284936 | 6895 |
| total_reward | 142.138440 | 189.626498 | -269.472433 | 4442.505616 | 6895 |
| velocity_penalty | -8.614789 | 8.614789 | -102.215058 | -2.518789 | 6895 |
