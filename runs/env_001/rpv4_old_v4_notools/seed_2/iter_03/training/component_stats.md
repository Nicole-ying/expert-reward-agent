# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_guidance | 0.038647 | 0.038647 | 0.120923 | 0.319601 | 0.319601 | 0.000000 | 0.599898 | 1003520 |
| component.orientation_penalty | -0.002985 | 0.002985 | 1.000000 | -0.002985 | 0.002985 | -1.131441 | -0.000000 | 1003520 |
| component.progress | 0.001590 | 0.001840 | 0.999992 | 0.001590 | 0.001840 | -0.015991 | 0.018783 | 1003520 |
| component.total_reward | 0.036084 | 0.041773 | 1.000000 | 0.036084 | 0.041773 | -1.142907 | 0.600037 | 1003520 |
| component.velocity_penalty | -0.001168 | 0.001168 | 0.999715 | -0.001169 | 0.001169 | -0.094896 | -0.000000 | 1003520 |
| generated_reward | 0.036084 | 0.041773 | 1.000000 | 0.036084 | 0.041773 | -1.142907 | 0.600037 | 1003520 |
| original_env_reward | -0.101691 | 4.221229 | 1.000000 | -0.101691 | 4.221229 | -100.000000 | 123.687054 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_guidance | 13.377216 | 13.377216 | 0.000000 | 69.810966 | 2894 |
| orientation_penalty | -1.033574 | 1.033574 | -19.683829 | -0.015299 | 2894 |
| progress | 0.550564 | 0.550564 | 0.065651 | 0.709748 | 2894 |
| total_reward | 12.489306 | 13.265770 | -7.382865 | 69.330392 | 2894 |
| velocity_penalty | -0.404901 | 0.404901 | -1.842818 | -0.075382 | 2894 |
