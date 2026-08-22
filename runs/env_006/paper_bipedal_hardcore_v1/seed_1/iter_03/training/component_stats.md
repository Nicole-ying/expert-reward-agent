# Reward Component Training Statistics

- steps_seen: 2000000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_efficiency_penalty | -0.016938 | 0.016938 | 1.000000 | -0.016938 | 0.016938 | -0.040000 | -0.000008 | 2000000 |
| component.angular_velocity_penalty | -0.000189 | 0.000189 | 0.999043 | -0.000189 | 0.000189 | -0.009252 | -0.000000 | 2000000 |
| component.progress_reward | 0.266792 | 0.269741 | 0.984274 | 0.271055 | 0.274051 | -0.428229 | 0.879704 | 2000000 |
| component.total_reward | 0.249665 | 0.256748 | 1.000000 | 0.249665 | 0.256748 | -0.444345 | 0.868355 | 2000000 |
| generated_reward | 0.249665 | 0.256748 | 1.000000 | 0.249665 | 0.256748 | -0.444345 | 0.868355 | 2000000 |
| original_env_reward | -0.436468 | 0.747407 | 1.000000 | -0.436468 | 0.747407 | -100.000000 | 0.942326 | 2000000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_efficiency_penalty | -2.973712 | 2.973712 | -33.663900 | -0.696233 | 11384 |
| angular_velocity_penalty | -0.033155 | 0.033155 | -0.285015 | -0.006121 | 11384 |
| progress_reward | 46.813298 | 46.818478 | -6.173047 | 493.832599 | 11384 |
| total_reward | 43.806431 | 43.995691 | -34.491833 | 475.818801 | 11384 |
