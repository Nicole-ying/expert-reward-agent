# Reward Component Training Statistics

- steps_seen: 1440000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.035393 | 0.035393 | 1.000000 | -0.035393 | 0.035393 | -0.080000 | -0.000014 | 1440000 |
| component.forward_progress | 0.217510 | 0.225610 | 1.000000 | 0.217510 | 0.225610 | -0.498432 | 0.778198 | 1440000 |
| component.stability_angle_penalty | -0.011210 | 0.011210 | 0.075350 | -0.148766 | 0.148766 | -34.559944 | -0.000000 | 1440000 |
| component.total_reward | 0.170987 | 0.209771 | 1.000000 | 0.170987 | 0.209771 | -20.000000 | 0.752483 | 1440000 |
| generated_reward | 0.170987 | 0.209771 | 1.000000 | 0.170987 | 0.209771 | -20.000000 | 0.752483 | 1440000 |
| original_env_reward | -0.489822 | 0.724772 | 1.000000 | -0.489822 | 0.724772 | -100.000000 | 0.842228 | 1440000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -6.142443 | 6.142443 | -86.476876 | -1.143450 | 8280 |
| forward_progress | 37.790241 | 37.908509 | -22.328263 | 305.164223 | 8280 |
| stability_angle_penalty | -1.949117 | 1.949117 | -569.141842 | 0.000000 | 8280 |
| total_reward | 29.712455 | 33.233275 | -484.349069 | 277.407935 | 8280 |
