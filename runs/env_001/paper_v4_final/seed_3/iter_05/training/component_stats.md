# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.002434 | 0.002434 | 0.921917 | -0.002640 | 0.002640 | -2.686142 | -0.000000 | 1003520 |
| component.contact_landing_reward | 2.150756 | 2.150756 | 0.541596 | 3.971148 | 3.971148 | 0.000000 | 4.999444 | 1003520 |
| component.lateral_pos_penalty | -0.016866 | 0.016866 | 0.999987 | -0.016866 | 0.016866 | -0.157053 | -0.000000 | 1003520 |
| component.progress_gated | 0.025642 | 0.031052 | 0.999387 | 0.025657 | 0.031071 | -0.211843 | 0.342883 | 1003520 |
| component.total_reward | 2.157098 | 2.168374 | 1.000000 | 2.157098 | 2.168374 | -2.402464 | 5.017947 | 1003520 |
| generated_reward | 2.157098 | 2.168374 | 1.000000 | 2.157098 | 2.168374 | -2.402464 | 5.017947 | 1003520 |
| original_env_reward | -0.116458 | 1.755948 | 1.000000 | -0.116458 | 1.755948 | -100.000000 | 129.265400 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.774993 | 0.774993 | -8.689542 | -0.005364 | 3150 |
| contact_landing_reward | 683.770618 | 683.770618 | 0.000000 | 4156.159623 | 3150 |
| lateral_pos_penalty | -5.370499 | 5.370499 | -109.504685 | -0.000037 | 3150 |
| progress_gated | 8.156448 | 8.164122 | -3.306182 | 13.551256 | 3150 |
| total_reward | 685.781573 | 685.983161 | -16.689104 | 4166.928198 | 3150 |
