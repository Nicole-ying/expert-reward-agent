# Reward Component Training Statistics

- steps_seen: 2080000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.gated_forward_speed | 0.163930 | 0.163930 | 0.992135 | 0.165229 | 0.165229 | 0.000000 | 1.635851 | 2080000 |
| component.stability_quad_penalty | -0.041868 | 0.041868 | 1.000000 | -0.041868 | 0.041868 | -17.218718 | -0.000000 | 2080000 |
| component.total_reward | 0.122062 | 0.170953 | 1.000000 | 0.122062 | 0.170953 | -17.218718 | 1.617024 | 2080000 |
| generated_reward | 0.122062 | 0.170953 | 1.000000 | 0.122062 | 0.170953 | -17.218718 | 1.617024 | 2080000 |
| original_env_reward | -0.502268 | 0.698005 | 1.000000 | -0.502268 | 0.698005 | -100.000000 | 1.089732 | 2080000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| gated_forward_speed | 28.461853 | 28.461853 | 0.010375 | 127.054276 | 11973 |
| stability_quad_penalty | -7.272377 | 7.272377 | -763.820084 | -0.574781 | 11973 |
| total_reward | 21.189475 | 25.498235 | -761.762013 | 113.765606 | 11973 |
