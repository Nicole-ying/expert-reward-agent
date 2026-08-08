# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.043644 | 0.043644 | 1.000000 | -0.043644 | 0.043644 | -0.080000 | -0.001082 | 1001472 |
| component.forward_gated | 0.504638 | 0.664861 | 0.628416 | 0.803032 | 1.057995 | -5.123003 | 6.056822 | 1001472 |
| component.height_reward | -0.036501 | 0.036501 | 0.259001 | -0.140931 | 0.140931 | -0.499994 | -0.000000 | 1001472 |
| component.total_reward | 0.424493 | 0.695347 | 1.000000 | 0.424493 | 0.695347 | -5.154307 | 6.014272 | 1001472 |
| generated_reward | 0.424493 | 0.695347 | 1.000000 | 0.424493 | 0.695347 | -5.154307 | 6.014272 | 1001472 |
| original_env_reward | -0.654410 | 1.087199 | 1.000000 | -0.654410 | 1.087199 | -6.272855 | 5.128697 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -5.743291 | 5.743291 | -47.196495 | -0.231772 | 7610 |
| forward_gated | 66.405159 | 68.972747 | -95.209161 | 1249.114231 | 7610 |
| height_reward | -4.803540 | 4.803540 | -115.515713 | 0.000000 | 7610 |
| total_reward | 55.858328 | 66.039135 | -189.680463 | 1201.283548 | 7610 |
