# Reward Component Training Statistics

- steps_seen: 5046272
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.002033 | 0.002033 | 1.000000 | -0.002033 | 0.002033 | -0.004000 | -0.000001 | 5046272 |
| component.angular_vel_penalty | -0.000017 | 0.000017 | 0.998929 | -0.000017 | 0.000017 | -0.000612 | -0.000000 | 5046272 |
| component.forward_progress | 0.256750 | 0.256750 | 0.954191 | 0.269076 | 0.269076 | 0.000000 | 0.704924 | 5046272 |
| component.posture_penalty | -0.003420 | 0.003420 | 0.999987 | -0.003420 | 0.003420 | -1.020414 | -0.000000 | 5046272 |
| component.total_reward | 0.251279 | 0.252560 | 1.000000 | 0.251279 | 0.252560 | -1.022948 | 0.703448 | 5046272 |
| generated_reward | 0.251279 | 0.252560 | 1.000000 | 0.251279 | 0.252560 | -1.022948 | 0.703448 | 5046272 |
| original_env_reward | -0.003042 | 0.304629 | 1.000000 | -0.003042 | 0.304629 | -100.000000 | 0.892057 | 5046272 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -1.112964 | 1.112964 | -3.404200 | -0.068888 | 9176 |
| angular_vel_penalty | -0.009568 | 0.009568 | -0.037944 | -0.000479 | 9176 |
| forward_progress | 140.334792 | 140.334792 | 0.000000 | 468.437287 | 9176 |
| posture_penalty | -1.876617 | 1.876617 | -35.800413 | -0.058482 | 9176 |
| total_reward | 137.335643 | 137.625466 | -28.181696 | 464.382022 | 9176 |
