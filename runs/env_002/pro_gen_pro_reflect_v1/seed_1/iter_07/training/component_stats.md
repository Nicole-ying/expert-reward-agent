# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.008196 | 0.008196 | 1.000000 | -0.008196 | 0.008196 | -0.020000 | -0.000007 | 1015808 |
| component.total_reward | 0.114447 | 0.119913 | 1.000000 | 0.114447 | 0.119913 | -0.020000 | 0.709290 | 1015808 |
| component.velocity_squared_gated | 0.122643 | 0.122643 | 0.999988 | 0.122644 | 0.122644 | 0.000000 | 0.720474 | 1015808 |
| generated_reward | 0.114447 | 0.119913 | 1.000000 | 0.114447 | 0.119913 | -0.020000 | 0.709290 | 1015808 |
| original_env_reward | -0.624353 | 0.936951 | 1.000000 | -0.624353 | 0.936951 | -100.000000 | 0.671511 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -1.088304 | 1.088304 | -16.838544 | -0.284863 | 7640 |
| total_reward | 15.181600 | 15.614019 | -15.276445 | 175.118708 | 7640 |
| velocity_squared_gated | 16.269904 | 16.269904 | 0.002118 | 180.417557 | 7640 |
