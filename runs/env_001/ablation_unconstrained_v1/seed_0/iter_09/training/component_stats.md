# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_proxy | 10.201154 | 10.201154 | 0.576766 | 17.686822 | 17.686822 | 0.000000 | 44.938211 | 1003520 |
| component.progress_reward | 0.004608 | 0.006050 | 0.999897 | 0.004609 | 0.006050 | -0.053489 | 0.076984 | 1003520 |
| component.proximity_reward | 0.333254 | 0.333254 | 1.000000 | 0.333254 | 0.333254 | 0.065496 | 0.499945 | 1003520 |
| component.stability_penalty | -0.021230 | 0.021230 | 1.000000 | -0.021230 | 0.021230 | -0.577044 | -0.000001 | 1003520 |
| component.total_reward | 6.954975 | 6.955020 | 1.000000 | 6.954975 | 6.955020 | -0.337993 | 20.000000 | 1003520 |
| generated_reward | 6.954975 | 6.955020 | 1.000000 | 6.954975 | 6.955020 | -0.337993 | 20.000000 | 1003520 |
| original_env_reward | -0.075175 | 2.012179 | 1.000000 | -0.075175 | 2.012179 | -100.000000 | 139.475452 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_proxy | 4568.032262 | 4568.032262 | 0.000000 | 32206.919598 | 2241 |
| progress_reward | 2.061002 | 2.152722 | -9.592239 | 2.833364 | 2241 |
| proximity_reward | 149.166763 | 149.166763 | 14.946080 | 454.353618 | 2241 |
| stability_penalty | -9.499985 | 9.499985 | -46.357649 | -1.062498 | 2241 |
| total_reward | 3114.332369 | 3114.332369 | 9.696213 | 16087.760226 | 2241 |
