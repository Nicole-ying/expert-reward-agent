# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.crash_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1003520 |
| component.progress_reward | 0.028523 | 0.031823 | 0.999691 | 0.028532 | 0.031833 | -0.156497 | 0.194675 | 1003520 |
| component.total_reward | 6.272365 | 6.274875 | 0.999991 | 6.272421 | 6.274931 | -0.149259 | 20.000000 | 1003520 |
| component.touchdown_bonus | 62.435427 | 62.435427 | 0.312177 | 200.000000 | 200.000000 | 0.000000 | 200.000000 | 1003520 |
| generated_reward | 6.272365 | 6.274875 | 0.999991 | 6.272421 | 6.274931 | -0.149259 | 20.000000 | 1003520 |
| original_env_reward | -0.243496 | 2.134837 | 1.000000 | -0.243496 | 2.134837 | -100.000000 | 150.191752 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| crash_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 5464 |
| progress_reward | 5.234341 | 5.234341 | 0.220644 | 7.079076 | 5464 |
| total_reward | 1150.369688 | 1150.369688 | 0.220644 | 17584.927354 | 5464 |
| touchdown_bonus | 11450.805271 | 11450.805271 | 0.000000 | 175800.000000 | 5464 |
