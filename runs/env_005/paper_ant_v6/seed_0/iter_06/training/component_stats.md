# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.327669 | 0.327669 | 1.000000 | -0.327669 | 0.327669 | -0.800000 | -0.001516 | 1001472 |
| component.descend_penalty | -0.036368 | 0.036368 | 0.227859 | -0.159610 | 0.159610 | -0.646591 | -0.000000 | 1001472 |
| component.forward_reward | 0.259119 | 0.259119 | 0.591086 | 0.438377 | 0.438377 | 0.000000 | 4.924766 | 1001472 |
| component.lateral_penalty | -0.184299 | 0.184299 | 0.999996 | -0.184300 | 0.184300 | -12.285845 | -0.000000 | 1001472 |
| component.total_reward | -0.289217 | 0.544828 | 1.000000 | -0.289217 | 0.544828 | -12.725270 | 4.413989 | 1001472 |
| generated_reward | -0.289217 | 0.544828 | 1.000000 | -0.289217 | 0.544828 | -12.725270 | 4.413989 | 1001472 |
| original_env_reward | -0.334869 | 1.030399 | 1.000000 | -0.334869 | 1.030399 | -6.327291 | 6.242991 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -8.623109 | 8.623109 | -417.410206 | -2.025737 | 38052 |
| descend_penalty | -0.957063 | 0.957063 | -13.715383 | 0.000000 | 38052 |
| forward_reward | 6.819403 | 6.819403 | 0.000000 | 76.318557 | 38052 |
| lateral_penalty | -4.850037 | 4.850037 | -226.063720 | -0.023976 | 38052 |
| total_reward | -7.610806 | 10.811912 | -621.947298 | 41.407936 | 38052 |
