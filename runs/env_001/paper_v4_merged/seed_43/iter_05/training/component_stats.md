# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.002402 | 0.002402 | 0.240244 | -0.010000 | 0.010000 | -0.010000 | -0.000000 | 1003520 |
| component.landing_contact_reward | 0.007004 | 0.007004 | 0.067151 | 0.104302 | 0.104302 | 0.000000 | 0.199852 | 1003520 |
| component.progress_shaping | 0.014114 | 0.015738 | 1.000000 | 0.014114 | 0.015738 | -0.067125 | 1.092923 | 1003520 |
| component.shaped_progress | 0.010783 | 0.012004 | 1.000000 | 0.010783 | 0.012004 | -0.063813 | 1.092923 | 1003520 |
| component.total_reward | 0.015384 | 0.017555 | 1.000000 | 0.015384 | 0.017555 | -0.073813 | 1.143416 | 1003520 |
| generated_reward | 0.015384 | 0.017555 | 1.000000 | 0.015384 | 0.017555 | -0.073813 | 1.143416 | 1003520 |
| original_env_reward | -1.420033 | 2.857509 | 1.000000 | -1.420033 | 2.857509 | -100.000000 | 151.824949 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.191095 | 0.191095 | -8.540000 | 0.000000 | 12557 |
| landing_contact_reward | 0.545845 | 0.545845 | 0.000000 | 128.888826 | 12557 |
| progress_shaping | 1.127543 | 1.128135 | -0.595622 | 1.711629 | 12557 |
| shaped_progress | 0.861374 | 0.861965 | -0.595622 | 1.558004 | 12557 |
| total_reward | 1.216124 | 1.219718 | -1.425622 | 122.968168 | 12557 |
