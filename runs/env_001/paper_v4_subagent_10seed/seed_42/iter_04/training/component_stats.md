# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.000488 | 0.000488 | 0.999750 | -0.000488 | 0.000488 | -0.304657 | -0.000000 | 1003520 |
| component.safe_progress | 0.004945 | 0.004945 | 0.738280 | 0.006698 | 0.006698 | 0.000000 | 0.028190 | 1003520 |
| component.soft_landing | 0.372824 | 0.372824 | 0.422332 | 0.882774 | 0.882774 | 0.000000 | 1.000000 | 1003520 |
| component.total_reward | 0.750105 | 0.750444 | 1.000000 | 0.750105 | 0.750444 | -0.302551 | 2.000326 | 1003520 |
| generated_reward | 0.750105 | 0.750444 | 1.000000 | 0.750105 | 0.750444 | -0.302551 | 2.000326 | 1003520 |
| original_env_reward | -0.168512 | 1.761824 | 1.000000 | -0.168512 | 1.761824 | -100.000000 | 136.378241 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -0.104455 | 0.104455 | -1.885103 | -0.000403 | 4685 |
| safe_progress | 1.057952 | 1.057952 | 0.100698 | 2.009911 | 4685 |
| soft_landing | 79.534859 | 79.534859 | 0.000000 | 869.769612 | 4685 |
| total_reward | 160.023216 | 160.027073 | -1.244071 | 1740.820258 | 4685 |
