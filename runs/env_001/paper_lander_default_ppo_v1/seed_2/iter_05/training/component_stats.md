# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.004694 | 0.005129 | 0.987486 | 0.004753 | 0.005194 | -0.059655 | 0.070767 | 1001472 |
| component.landing_approach | 0.222631 | 0.222631 | 0.989629 | 0.224964 | 0.224964 | 0.000000 | 0.299949 | 1001472 |
| component.stability_penalty | -0.002295 | 0.002295 | 1.000000 | -0.002295 | 0.002295 | -0.088718 | -0.000000 | 1001472 |
| component.time_penalty | -0.020000 | 0.020000 | 1.000000 | -0.020000 | 0.020000 | -0.020000 | -0.020000 | 1001472 |
| component.total_reward | 0.205030 | 0.205377 | 1.000000 | 0.205030 | 0.205377 | -0.155878 | 0.279949 | 1001472 |
| generated_reward | 0.205030 | 0.205377 | 1.000000 | 0.205030 | 0.205377 | -0.155878 | 0.279949 | 1001472 |
| original_env_reward | 0.135861 | 0.938947 | 1.000000 | 0.135861 | 0.938947 | -100.000000 | 116.622276 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 2.466199 | 2.466199 | 0.280340 | 2.840322 | 1905 |
| landing_approach | 117.009690 | 117.009690 | 1.079248 | 278.315074 | 1905 |
| stability_penalty | -1.205830 | 1.205830 | -4.782244 | -0.576079 | 1905 |
| time_penalty | -10.507087 | 10.507087 | -20.000000 | -1.180000 | 1905 |
| total_reward | 107.762972 | 107.784807 | -2.818070 | 260.252667 | 1905 |
