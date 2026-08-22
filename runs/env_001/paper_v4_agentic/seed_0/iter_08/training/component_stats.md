# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_bonus | 0.400491 | 0.400491 | 1.000000 | 0.400491 | 0.400491 | 0.000171 | 0.992670 | 1003520 |
| component.landing_safety_penalty | 0.002210 | 0.002210 | 1.000000 | 0.002210 | 0.002210 | 0.000000 | 0.421186 | 1003520 |
| component.progress_reward | 0.001967 | 0.002260 | 0.998362 | 0.001970 | 0.002264 | -0.022907 | 0.039964 | 1003520 |
| component.total_reward | 0.400248 | 0.400849 | 1.000000 | 0.400248 | 0.400849 | -0.400605 | 0.992639 | 1003520 |
| generated_reward | 0.400248 | 0.400849 | 1.000000 | 0.400248 | 0.400849 | -0.400605 | 0.992639 | 1003520 |
| original_env_reward | 0.070998 | 1.129353 | 1.000000 | 0.070998 | 1.129353 | -100.000000 | 104.300021 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_bonus | 241.057159 | 241.057159 | 0.094007 | 795.840555 | 1662 |
| landing_safety_penalty | 1.333152 | 1.333152 | 0.158507 | 14.061094 | 1662 |
| progress_reward | 1.185256 | 1.233808 | -7.492698 | 1.419474 | 1662 |
| total_reward | 240.909263 | 241.047021 | -5.508691 | 796.735146 | 1662 |
