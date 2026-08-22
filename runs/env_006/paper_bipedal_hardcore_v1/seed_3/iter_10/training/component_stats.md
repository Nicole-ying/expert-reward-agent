# Reward Component Training Statistics

- steps_seen: 1360000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.air_penalty | -0.074561 | 0.074685 | 0.984095 | -0.075766 | 0.075893 | -0.281772 | 0.159618 | 1360000 |
| component.progress_gated | 0.177926 | 0.177926 | 0.930249 | 0.191267 | 0.191267 | 0.000000 | 0.718600 | 1360000 |
| component.total_reward | 0.103365 | 0.146999 | 0.999188 | 0.103449 | 0.147119 | -0.279827 | 0.688168 | 1360000 |
| generated_reward | 0.103365 | 0.146999 | 0.999188 | 0.103449 | 0.147119 | -0.279827 | 0.688168 | 1360000 |
| original_env_reward | -1.052247 | 1.317922 | 1.000000 | -1.052247 | 1.317922 | -100.000000 | 0.807334 | 1360000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| air_penalty | -6.545986 | 6.545986 | -125.424881 | -2.220589 | 15484 |
| progress_gated | 15.617659 | 15.617659 | 0.000638 | 103.750128 | 15484 |
| total_reward | 9.071673 | 10.334770 | -110.754793 | 73.231363 | 15484 |
