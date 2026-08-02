# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.074159 | 0.074159 | 1.000000 | -0.074159 | 0.074159 | -178.427048 | -0.000000 | 1003520 |
| component.proximity_delta | 0.788948 | 0.836333 | 0.999995 | 0.788952 | 0.836337 | -1.967766 | 2.125652 | 1003520 |
| component.soft_approach_bonus | 0.014687 | 0.014687 | 0.001980 | 7.417386 | 7.417386 | 0.000000 | 9.979495 | 1003520 |
| component.total_reward | 0.621458 | 0.776420 | 1.000000 | 0.621458 | 0.776420 | -20.000000 | 9.988721 | 1003520 |
| component.velocity_danger | -0.120503 | 0.120503 | 1.000000 | -0.120503 | 0.120503 | -0.691575 | -0.000000 | 1003520 |
| generated_reward | 0.621458 | 0.776420 | 1.000000 | 0.621458 | 0.776420 | -20.000000 | 9.988721 | 1003520 |
| original_env_reward | -1.486002 | 2.428149 | 1.000000 | -1.486002 | 2.428149 | -100.000000 | 140.491361 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -5.283991 | 5.283991 | -1441.987472 | -0.011275 | 14084 |
| proximity_delta | 56.206520 | 56.206520 | 1.938075 | 70.452342 | 14084 |
| soft_approach_bonus | 1.046460 | 1.046460 | 0.000000 | 8032.030212 | 14084 |
| total_reward | 44.273563 | 48.820702 | -996.782460 | 7785.184649 | 14084 |
| velocity_danger | -8.585059 | 8.585059 | -15.227726 | -4.957818 | 14084 |
