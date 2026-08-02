# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_bonus | 0.657256 | 0.657256 | 1.000000 | 0.657256 | 0.657256 | 0.000117 | 1.990817 | 1003520 |
| component.landing_safety_penalty | 0.000733 | 0.000733 | 1.000000 | 0.000733 | 0.000733 | 0.000000 | 0.129328 | 1003520 |
| component.progress_reward | 0.001999 | 0.002395 | 0.996266 | 0.002007 | 0.002404 | -0.038630 | 0.038492 | 1003520 |
| component.total_reward | 0.658522 | 0.658665 | 1.000000 | 0.658522 | 0.658665 | -0.103781 | 1.990800 | 1003520 |
| generated_reward | 0.658522 | 0.658665 | 1.000000 | 0.658522 | 0.658665 | -0.103781 | 1.990800 | 1003520 |
| original_env_reward | 0.060811 | 1.234872 | 1.000000 | 0.060811 | 1.234872 | -100.000000 | 125.066134 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_bonus | 389.648348 | 389.648348 | 0.188014 | 1591.052300 | 1691 |
| landing_safety_penalty | 0.434824 | 0.434824 | 0.064737 | 4.907322 | 1691 |
| progress_reward | 1.183935 | 1.254644 | -16.995535 | 1.421149 | 1691 |
| total_reward | 390.397459 | 390.435440 | -14.545705 | 1592.324672 | 1691 |
