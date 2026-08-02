# Reward Component Training Statistics

- steps_seen: 1120000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.017624 | 0.017624 | 1.000000 | -0.017624 | 0.017624 | -0.040000 | -0.000001 | 1120000 |
| component.hinge_balance_penalty | -0.002731 | 0.002731 | 0.030522 | -0.089487 | 0.089487 | -0.963701 | -0.000000 | 1120000 |
| component.progress | 0.181984 | 0.181984 | 0.884918 | 0.205650 | 0.205650 | 0.000000 | 0.748939 | 1120000 |
| component.total_reward | 0.161628 | 0.172755 | 1.000000 | 0.161628 | 0.172755 | -0.979985 | 0.732065 | 1120000 |
| generated_reward | 0.161628 | 0.172755 | 1.000000 | 0.161628 | 0.172755 | -0.979985 | 0.732065 | 1120000 |
| original_env_reward | -0.515209 | 0.728003 | 1.000000 | -0.515209 | 0.728003 | -100.000000 | 0.731876 | 1120000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -2.959506 | 2.959506 | -33.603633 | -0.719902 | 6665 |
| hinge_balance_penalty | -0.458983 | 0.458983 | -67.700286 | 0.000000 | 6665 |
| progress | 30.543904 | 30.543904 | 0.000000 | 265.234024 | 6665 |
| total_reward | 27.125414 | 27.712472 | -79.929096 | 250.898798 | 6665 |
