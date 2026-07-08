# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.005078 | 0.005461 | 0.995256 | 0.005102 | 0.005487 | -0.032306 | 0.038742 | 1003520 |
| component.soft_landing_proxy | 1.727730 | 1.727730 | 0.434542 | 3.975975 | 3.975975 | 0.000000 | 4.986024 | 1003520 |
| component.stability_penalty | -0.001527 | 0.001527 | 1.000000 | -0.001527 | 0.001527 | -0.068670 | -0.000000 | 1003520 |
| component.total_reward | 1.731280 | 1.731993 | 1.000000 | 1.731280 | 1.731993 | -0.095010 | 4.986021 | 1003520 |
| generated_reward | 1.731280 | 1.731993 | 1.000000 | 1.731280 | 1.731993 | -0.095010 | 4.986021 | 1003520 |
| original_env_reward | -0.139663 | 1.617702 | 1.000000 | -0.139663 | 1.617702 | -100.000000 | 129.957835 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 1.185373 | 1.185424 | -0.109045 | 1.417501 | 4294 |
| soft_landing_proxy | 402.096597 | 402.096597 | 0.000000 | 4284.573868 | 4294 |
| stability_penalty | -0.356481 | 0.356481 | -3.288328 | -0.093437 | 4294 |
| total_reward | 402.925489 | 402.943685 | -1.373999 | 4285.726248 | 4294 |
