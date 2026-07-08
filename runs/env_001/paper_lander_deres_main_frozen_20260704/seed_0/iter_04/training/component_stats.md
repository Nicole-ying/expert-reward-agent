# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.soft_landing_proxy | 0.251996 | 0.251996 | 0.503992 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 1003520 |
| component.stability_penalty | -0.002171 | 0.002171 | 0.999987 | -0.002171 | 0.002171 | -0.367833 | -0.000000 | 1003520 |
| component.total_reward | 0.331048 | 0.331814 | 1.000000 | 0.331048 | 0.331814 | -0.292920 | 0.793378 | 1003520 |
| component.velocity_alignment_reward | 0.081223 | 0.081223 | 0.651267 | 0.124716 | 0.124716 | 0.000000 | 0.800000 | 1003520 |
| generated_reward | 0.331048 | 0.331814 | 1.000000 | 0.331048 | 0.331814 | -0.292920 | 0.793378 | 1003520 |
| original_env_reward | -0.028022 | 1.365994 | 1.000000 | -0.028022 | 1.365994 | -100.000000 | 118.882823 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| soft_landing_proxy | 108.049743 | 108.049743 | 0.000000 | 425.500000 | 2332 |
| stability_penalty | -0.933337 | 0.933337 | -4.366710 | -0.215166 | 2332 |
| total_reward | 142.006912 | 142.008390 | -1.723260 | 459.205771 | 2332 |
| velocity_alignment_reward | 34.890506 | 34.890506 | 0.000000 | 49.864129 | 2332 |
