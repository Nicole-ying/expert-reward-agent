# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_bonus | 0.001484 | 0.001484 | 0.001484 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1003520 |
| component.progress_reward | 0.116703 | 0.123217 | 0.999996 | 0.116704 | 0.123217 | -0.316916 | 0.399409 | 1003520 |
| component.stability_penalty | -0.033049 | 0.033049 | 1.000000 | -0.033049 | 0.033049 | -0.285705 | -0.000000 | 1003520 |
| component.total_reward | 0.085138 | 0.097445 | 1.000000 | 0.085138 | 0.097445 | -0.484815 | 1.013919 | 1003520 |
| generated_reward | 0.085138 | 0.097445 | 1.000000 | 0.085138 | 0.097445 | -0.484815 | 1.013919 | 1003520 |
| original_env_reward | -0.183874 | 3.053349 | 1.000000 | -0.183874 | 3.053349 | -100.000000 | 130.124835 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 0.165334 | 0.165334 | 0.000000 | 1.000000 | 9006 |
| progress_reward | 13.001459 | 13.001809 | -1.577857 | 14.126441 | 9006 |
| stability_penalty | -3.681912 | 3.681912 | -12.486241 | -1.377383 | 9006 |
| total_reward | 9.484881 | 9.510885 | -9.875249 | 13.238822 | 9006 |
