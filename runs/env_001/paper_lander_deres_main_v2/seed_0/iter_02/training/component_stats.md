# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_delta | -0.001750 | 0.004481 | 1.000000 | -0.001750 | 0.004481 | -0.085173 | 0.051395 | 1003520 |
| component.soft_landing_proxy | 0.000004 | 0.000004 | 0.000008 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 1003520 |
| component.stability_penalty | -0.004066 | 0.004066 | 0.375214 | -0.010837 | 0.010837 | -0.602083 | -0.000000 | 1003520 |
| component.total_reward | -0.005813 | 0.007344 | 1.000000 | -0.005813 | 0.007344 | -0.596417 | 0.492739 | 1003520 |
| generated_reward | -0.005813 | 0.007344 | 1.000000 | -0.005813 | 0.007344 | -0.596417 | 0.492739 | 1003520 |
| original_env_reward | -5.398728 | 5.686685 | 1.000000 | -5.398728 | 5.686685 | -100.000000 | 135.725191 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_delta | -0.186139 | 0.238266 | -13.696415 | 1.412644 | 9433 |
| soft_landing_proxy | 0.000424 | 0.000424 | 0.000000 | 0.500000 | 9433 |
| stability_penalty | -0.432540 | 0.432540 | -19.582841 | -0.008365 | 9433 |
| total_reward | -0.618255 | 0.618255 | -18.488694 | -0.080302 | 9433 |
