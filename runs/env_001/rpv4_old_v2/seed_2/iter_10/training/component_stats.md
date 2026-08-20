# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_distance_progress | 0.014360 | 0.016325 | 0.995347 | 0.014427 | 0.016401 | -0.146868 | 0.187610 | 1003520 |
| component.landing_contact_success | 1.552255 | 1.552255 | 0.542651 | 2.860504 | 2.860504 | 0.000000 | 4.992964 | 1003520 |
| component.orientation_penalty | -0.013478 | 0.013478 | 0.999854 | -0.013480 | 0.013480 | -7.973717 | -0.000000 | 1003520 |
| component.soft_landing | -0.022843 | 0.022843 | 0.997767 | -0.022894 | 0.022894 | -3.171501 | -0.000000 | 1003520 |
| component.thrust_penalty | -0.030336 | 0.030336 | 0.606716 | -0.050000 | 0.050000 | -0.050000 | -0.000000 | 1003520 |
| component.total_reward | 1.499958 | 1.576966 | 1.000000 | 1.499958 | 1.576966 | -8.389984 | 4.992966 | 1003520 |
| generated_reward | 1.499958 | 1.576966 | 1.000000 | 1.499958 | 1.576966 | -8.389984 | 4.992966 | 1003520 |
| original_env_reward | 0.003654 | 1.434979 | 1.000000 | 0.003654 | 1.434979 | -100.000000 | 121.494095 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_distance_progress | 5.569199 | 5.569199 | 0.311162 | 7.099434 | 2584 |
| landing_contact_success | 602.518778 | 602.518778 | 0.000000 | 3818.242088 | 2584 |
| orientation_penalty | -5.232538 | 5.232538 | -111.249927 | -0.010145 | 2584 |
| soft_landing | -8.866057 | 8.866057 | -59.218245 | -1.918484 | 2584 |
| thrust_penalty | -11.769060 | 11.769060 | -41.550000 | -1.700000 | 2584 |
| total_reward | 582.220323 | 599.635680 | -136.005989 | 3789.417751 | 2584 |
