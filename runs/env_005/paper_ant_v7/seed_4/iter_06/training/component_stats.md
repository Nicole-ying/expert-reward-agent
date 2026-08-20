# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angular_velocity_penalty | -0.127662 | 0.127662 | 1.000000 | -0.127662 | 0.127662 | -3.097635 | -0.000000 | 1001472 |
| component.gated_forward | 0.750656 | 0.955731 | 1.000000 | 0.750656 | 0.955731 | -7.279479 | 9.370040 | 1001472 |
| component.height_hinge_penalty | -0.151017 | 0.151017 | 0.386381 | -0.390850 | 0.390850 | -3.488323 | -0.000000 | 1001472 |
| component.lateral_penalty | -0.176576 | 0.176576 | 0.999995 | -0.176577 | 0.176577 | -10.343726 | -0.000000 | 1001472 |
| component.total_reward | 0.295401 | 1.054012 | 1.000000 | 0.295401 | 1.054012 | -12.015556 | 8.857108 | 1001472 |
| generated_reward | 0.295401 | 1.054012 | 1.000000 | 0.295401 | 1.054012 | -12.015556 | 8.857108 | 1001472 |
| original_env_reward | -0.850238 | 1.047652 | 1.000000 | -0.850238 | 1.047652 | -5.378823 | 3.788962 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angular_velocity_penalty | -28.932962 | 28.932962 | -190.065720 | -0.552261 | 4418 |
| gated_forward | 170.093763 | 177.048685 | -189.280826 | 2587.644503 | 4418 |
| height_hinge_penalty | -34.232552 | 34.232552 | -386.779241 | 0.000000 | 4418 |
| lateral_penalty | -40.017347 | 40.017347 | -397.695306 | -0.099181 | 4418 |
| total_reward | 66.910902 | 165.652983 | -653.814365 | 2172.609428 | 4418 |
