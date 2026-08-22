# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.006540 | 0.006540 | 0.102795 | -0.063619 | 0.063619 | -0.200000 | 0.000000 | 1003520 |
| component.landing_bonus | 0.004436 | 0.004436 | 0.014339 | 0.309357 | 0.309357 | 0.000000 | 44.546264 | 1003520 |
| component.orientation_penalty | -0.014985 | 0.014985 | 0.999999 | -0.014985 | 0.014985 | -3.681531 | -0.000000 | 1003520 |
| component.progress_reward | 0.193588 | 0.204822 | 0.999998 | 0.193589 | 0.204822 | -0.484716 | 0.508720 | 1003520 |
| component.soft_approach_reward | 0.072602 | 0.072602 | 1.000000 | 0.072602 | 0.072602 | 0.000065 | 0.394427 | 1003520 |
| component.total_reward | 0.248044 | 0.277398 | 1.000000 | 0.248044 | 0.277398 | -3.600943 | 20.000000 | 1003520 |
| generated_reward | 0.248044 | 0.277398 | 1.000000 | 0.248044 | 0.277398 | -3.600943 | 20.000000 | 1003520 |
| original_env_reward | -1.660712 | 2.351864 | 1.000000 | -1.660712 | 2.351864 | -100.000000 | 133.819538 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -0.457864 | 0.457864 | -11.600000 | 0.000000 | 14332 |
| landing_bonus | 0.310588 | 0.310588 | 0.000000 | 44.546264 | 14332 |
| orientation_penalty | -1.049263 | 1.049263 | -47.934567 | -0.000001 | 14332 |
| progress_reward | 13.554687 | 13.554951 | -1.893430 | 16.951731 | 14332 |
| soft_approach_reward | 5.082390 | 5.082390 | 1.069524 | 20.532678 | 14332 |
| total_reward | 17.366505 | 17.419244 | -27.482644 | 43.464437 | 14332 |
