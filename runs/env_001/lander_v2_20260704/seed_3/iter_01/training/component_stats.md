# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.015675 | 0.016653 | 0.999990 | 0.015675 | 0.016654 | -0.038337 | 0.041957 | 1003520 |
| component.soft_landing_proxy | 0.024883 | 0.024883 | 0.017711 | 1.404947 | 1.404947 | 0.000000 | 4.589896 | 1003520 |
| component.stability_penalty | -0.019087 | 0.019087 | 1.000000 | -0.019087 | 0.019087 | -0.589405 | -0.000000 | 1003520 |
| component.total_reward | 0.021470 | 0.032705 | 1.000000 | 0.021470 | 0.032705 | -0.609191 | 4.587505 | 1003520 |
| generated_reward | 0.021470 | 0.032705 | 1.000000 | 0.021470 | 0.032705 | -0.609191 | 4.587505 | 1003520 |
| original_env_reward | -1.313021 | 2.539410 | 1.000000 | -1.313021 | 2.539410 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 1.122460 | 1.122460 | 0.058931 | 1.400713 | 14013 |
| soft_landing_proxy | 1.781926 | 1.781926 | 0.000000 | 594.397963 | 14013 |
| stability_penalty | -1.366782 | 1.366782 | -16.298911 | -0.695030 | 14013 |
| total_reward | 1.537604 | 2.082023 | -15.416176 | 581.362580 | 14013 |
