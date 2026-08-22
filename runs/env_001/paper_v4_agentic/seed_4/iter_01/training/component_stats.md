# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_gated | 0.002130 | 0.002130 | 0.901430 | 0.002363 | 0.002363 | 0.000000 | 0.005918 | 1003520 |
| component.soft_landing | 0.008187 | 0.008187 | 0.007429 | 1.102064 | 1.102064 | 0.000000 | 1.828896 | 1003520 |
| component.total_reward | 0.010317 | 0.010317 | 0.908216 | 0.011360 | 0.011360 | 0.000000 | 1.828896 | 1003520 |
| generated_reward | 0.010317 | 0.010317 | 0.908216 | 0.011360 | 0.011360 | 0.000000 | 1.828896 | 1003520 |
| original_env_reward | -1.527477 | 2.489255 | 1.000000 | -1.527477 | 2.489255 | -100.000000 | 124.884751 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_gated | 0.152804 | 0.152804 | 0.000193 | 0.332771 | 13988 |
| soft_landing | 0.587352 | 0.587352 | 0.000000 | 823.967187 | 13988 |
| total_reward | 0.740156 | 0.740156 | 0.000193 | 824.299958 | 13988 |
