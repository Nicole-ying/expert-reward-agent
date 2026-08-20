# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.041633 | 0.041633 | 1.000000 | -0.041633 | 0.041633 | -0.080000 | -0.001411 | 1001472 |
| component.forward_gated | 0.307463 | 0.401874 | 0.445458 | 0.690217 | 0.902158 | -3.876956 | 6.198412 | 1001472 |
| component.lateral_penalty | -0.058058 | 0.058058 | 0.445457 | -0.130334 | 0.130334 | -4.736868 | -0.000000 | 1001472 |
| component.total_reward | 0.207772 | 0.392525 | 1.000000 | 0.207772 | 0.392525 | -5.282597 | 5.649820 | 1001472 |
| generated_reward | 0.207772 | 0.392525 | 1.000000 | 0.207772 | 0.392525 | -5.282597 | 5.649820 | 1001472 |
| original_env_reward | -0.745701 | 0.986196 | 1.000000 | -0.745701 | 0.986196 | -5.976644 | 4.861674 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -8.585653 | 8.585653 | -48.066213 | -0.293309 | 4851 |
| forward_gated | 63.214554 | 66.310962 | -148.455385 | 1403.207875 | 4851 |
| lateral_penalty | -11.965852 | 11.965852 | -224.871108 | -0.016098 | 4851 |
| total_reward | 42.663048 | 53.586876 | -247.713691 | 1252.619971 | 4851 |
