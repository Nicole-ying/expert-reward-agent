# Reward Component Training Statistics

- steps_seen: 1000448
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_stability_reward | 1.360007 | 1.363194 | 1.000000 | 1.360007 | 1.363194 | -0.619945 | 3.646188 | 1000448 |
| component.stability_penalty | -0.002537 | 0.002537 | 0.999999 | -0.002537 | 0.002537 | -0.976414 | -0.000000 | 1000448 |
| component.total_reward | 1.373597 | 1.377576 | 1.000000 | 1.373597 | 1.377576 | -0.790371 | 3.632745 | 1000448 |
| component.vertical_pushoff | 0.016128 | 0.016128 | 0.220005 | 0.073305 | 0.073305 | 0.000000 | 0.359875 | 1000448 |
| generated_reward | 1.373597 | 1.377576 | 1.000000 | 1.373597 | 1.377576 | -0.790371 | 3.632745 | 1000448 |
| original_env_reward | 2.403345 | 2.403748 | 1.000000 | 2.403345 | 2.403748 | -1.113671 | 4.656812 | 1000448 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_stability_reward | 158.291617 | 158.603120 | -13.648168 | 1312.157779 | 8594 |
| stability_penalty | -0.295294 | 0.295294 | -4.001844 | -0.025435 | 8594 |
| total_reward | 159.871669 | 160.271160 | -13.910367 | 1355.871406 | 8594 |
| vertical_pushoff | 1.875346 | 1.875346 | 0.000000 | 52.356379 | 8594 |
