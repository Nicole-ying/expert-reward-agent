# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.001832 | 0.002404 | 0.999094 | 0.001834 | 0.002406 | -0.027042 | 0.051090 | 1003520 |
| component.contact_bonus | 1.019197 | 1.019197 | 0.535094 | 1.904706 | 1.904706 | 0.000000 | 2.000000 | 1003520 |
| component.proximity_quality | 0.844682 | 0.844682 | 1.000000 | 0.844682 | 0.844682 | 0.010194 | 1.994381 | 1003520 |
| component.stability_penalty | -0.001626 | 0.001626 | 1.000000 | -0.001626 | 0.001626 | -0.058941 | -0.000000 | 1003520 |
| component.total_reward | 1.864086 | 1.864090 | 1.000000 | 1.864086 | 1.864090 | -0.054925 | 3.994378 | 1003520 |
| generated_reward | 1.864086 | 1.864090 | 1.000000 | 1.864086 | 1.864090 | -0.054925 | 3.994378 | 1003520 |
| original_env_reward | -0.002263 | 1.499564 | 1.000000 | -0.002263 | 1.499564 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 1.027825 | 1.080705 | -3.992921 | 1.420019 | 1784 |
| contact_bonus | 571.248879 | 571.248879 | 0.000000 | 1728.000000 | 1784 |
| proximity_quality | 473.090343 | 473.090343 | 4.329670 | 1678.542946 | 1784 |
| stability_penalty | -0.912956 | 0.912956 | -4.324819 | -0.118951 | 1784 |
| total_reward | 1044.454090 | 1044.454090 | 5.114385 | 3278.568923 | 1784 |
