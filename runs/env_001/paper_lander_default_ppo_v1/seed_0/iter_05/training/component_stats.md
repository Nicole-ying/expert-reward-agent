# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_contact_reward | 0.011204 | 0.011204 | 0.002241 | 5.000000 | 5.000000 | 0.000000 | 5.000000 | 1001472 |
| component.landing_shaping_reward | 0.343811 | 0.399378 | 1.000000 | 0.343811 | 0.399378 | -7.316586 | 34.175140 | 1001472 |
| component.progress_reward | 0.014973 | 0.015896 | 0.999474 | 0.014981 | 0.015904 | -0.041372 | 0.042519 | 1001472 |
| component.stability_penalty | -0.001273 | 0.001273 | 1.000000 | -0.001273 | 0.001273 | -0.005905 | -0.000000 | 1001472 |
| component.total_reward | 0.356083 | 0.407309 | 1.000000 | 0.356083 | 0.407309 | -7.359870 | 20.000000 | 1001472 |
| generated_reward | 0.356083 | 0.407309 | 1.000000 | 0.356083 | 0.407309 | -7.359870 | 20.000000 | 1001472 |
| original_env_reward | -0.968605 | 2.753531 | 1.000000 | -0.968605 | 2.753531 | -100.000000 | 119.335129 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_contact_reward | 0.878003 | 0.878003 | 0.000000 | 20.000000 | 12779 |
| landing_shaping_reward | 26.943498 | 26.955754 | -5.567271 | 71.646443 | 12779 |
| progress_reward | 1.173334 | 1.173334 | 0.049603 | 1.408428 | 12779 |
| stability_penalty | -0.099756 | 0.099756 | -0.358303 | -0.063805 | 12779 |
| total_reward | 27.905232 | 27.913652 | -5.180956 | 77.963607 | 12779 |
