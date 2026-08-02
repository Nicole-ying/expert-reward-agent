# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.001122 | 0.001122 | 0.999981 | -0.001122 | 0.001122 | -0.332400 | -0.000000 | 1003520 |
| component.progress_delta | 0.015804 | 0.015804 | 0.882661 | 0.017905 | 0.017905 | 0.000000 | 0.045713 | 1003520 |
| component.soft_landing | 0.013700 | 0.013700 | 0.026723 | 0.512669 | 0.512669 | 0.000000 | 1.000000 | 1003520 |
| component.total_reward | 0.028382 | 0.030100 | 0.999999 | 0.028382 | 0.030100 | -0.332400 | 1.000148 | 1003520 |
| generated_reward | 0.028382 | 0.030100 | 0.999999 | 0.028382 | 0.030100 | -0.332400 | 1.000148 | 1003520 |
| original_env_reward | -1.265046 | 2.646513 | 1.000000 | -1.265046 | 2.646513 | -100.000000 | 128.974186 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -0.086753 | 0.086753 | -2.604113 | -0.000326 | 12981 |
| progress_delta | 1.221652 | 1.221652 | 0.101378 | 3.298255 | 12981 |
| soft_landing | 1.059105 | 1.059105 | 0.000000 | 546.049235 | 12981 |
| total_reward | 2.194004 | 2.195477 | -1.121889 | 547.316633 | 12981 |
