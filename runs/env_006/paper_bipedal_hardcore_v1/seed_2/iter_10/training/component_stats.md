# Reward Component Training Statistics

- steps_seen: 1200000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.017472 | 0.017472 | 1.000000 | -0.017472 | 0.017472 | -0.040000 | -0.000008 | 1200000 |
| component.progress | 0.185846 | 0.185846 | 0.891317 | 0.208508 | 0.208508 | 0.000000 | 0.738602 | 1200000 |
| component.total_reward | 0.168375 | 0.174808 | 1.000000 | 0.168375 | 0.174808 | -0.040000 | 0.723199 | 1200000 |
| component.vertical_hinge_penalty | -0.000000 | 0.000000 | 0.000007 | -0.003254 | 0.003254 | -0.007849 | -0.000000 | 1200000 |
| generated_reward | 0.168375 | 0.174808 | 1.000000 | 0.168375 | 0.174808 | -0.040000 | 0.723199 | 1200000 |
| original_env_reward | -0.554667 | 0.777464 | 1.000000 | -0.554667 | 0.777464 | -100.000000 | 0.736663 | 1200000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -2.739972 | 2.739972 | -38.367006 | -0.662136 | 7639 |
| progress | 29.150823 | 29.150823 | 0.001344 | 242.359341 | 7639 |
| total_reward | 26.410848 | 26.575994 | -18.516132 | 229.180216 | 7639 |
| vertical_hinge_penalty | -0.000003 | 0.000003 | -0.018703 | 0.000000 | 7639 |
