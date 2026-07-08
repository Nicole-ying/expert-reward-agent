# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.027185 | 0.029664 | 0.996469 | 0.027281 | 0.029769 | -0.117043 | 0.137367 | 1003520 |
| component.soft_landing_proxy | 0.690356 | 0.690356 | 0.229476 | 3.008399 | 3.008399 | 0.000000 | 4.975485 | 1003520 |
| component.stability_penalty | -0.002257 | 0.002257 | 1.000000 | -0.002257 | 0.002257 | -0.058941 | -0.000000 | 1003520 |
| component.total_reward | 0.715284 | 0.717824 | 1.000000 | 0.715284 | 0.717824 | -0.128332 | 4.975485 | 1003520 |
| generated_reward | 0.715284 | 0.717824 | 1.000000 | 0.715284 | 0.717824 | -0.128332 | 4.975485 | 1003520 |
| original_env_reward | -0.578566 | 2.099724 | 1.000000 | -0.578566 | 2.099724 | -100.000000 | 138.679758 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 3.488938 | 3.491853 | -4.742023 | 4.246571 | 7815 |
| soft_landing_proxy | 88.284853 | 88.284853 | 0.000000 | 4263.342706 | 7815 |
| stability_penalty | -0.289404 | 0.289404 | -3.484893 | -0.087263 | 7815 |
| total_reward | 91.484387 | 91.486727 | -2.926354 | 4267.336789 | 7815 |
