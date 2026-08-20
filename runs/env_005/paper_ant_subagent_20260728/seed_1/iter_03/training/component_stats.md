# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_gated | 0.374131 | 0.597224 | 1.000000 | 0.374131 | 0.597224 | -5.146311 | 5.353976 | 1001472 |
| component.health_gate | 0.574500 | 0.574500 | 0.578783 | 0.992599 | 0.992599 | 0.000000 | 1.000000 | 1001472 |
| component.total_reward | 0.042811 | 0.875418 | 1.000000 | 0.042811 | 0.875418 | -5.146311 | 5.125606 | 1001472 |
| component.upright_penalty | -0.323284 | 0.323284 | 0.426373 | -0.758217 | 0.758217 | -0.800000 | -0.000000 | 1001472 |
| component.z_velocity_penalty | -0.008036 | 0.008036 | 0.073386 | -0.109509 | 0.109509 | -1.308921 | -0.000000 | 1001472 |
| generated_reward | 0.042811 | 0.875418 | 1.000000 | 0.042811 | 0.875418 | -5.146311 | 5.125606 | 1001472 |
| original_env_reward | -0.784092 | 1.080542 | 1.000000 | -0.784092 | 1.080542 | -6.737460 | 4.843324 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_gated | 45.780707 | 49.300917 | -113.325533 | 799.159732 | 8182 |
| health_gate | 70.307212 | 70.307212 | 6.000000 | 884.490795 | 8182 |
| total_reward | 5.285421 | 82.573289 | -776.497241 | 790.292602 | 8182 |
| upright_penalty | -39.511800 | 39.511800 | -766.557323 | 0.000000 | 8182 |
| z_velocity_penalty | -0.983486 | 0.983486 | -8.867130 | 0.000000 | 8182 |
