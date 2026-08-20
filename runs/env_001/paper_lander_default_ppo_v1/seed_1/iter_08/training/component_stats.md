# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.005751 | 0.005751 | 0.575135 | -0.010000 | 0.010000 | -0.010000 | 0.000000 | 1001472 |
| component.progress_reward | 0.003820 | 0.004465 | 0.984548 | 0.003880 | 0.004535 | -0.036708 | 0.039538 | 1001472 |
| component.soft_landing_proxy | 2.876003 | 2.876003 | 0.981541 | 2.930089 | 2.930089 | 0.000000 | 4.997604 | 1001472 |
| component.speed_tracking_reward | -0.317084 | 0.317084 | 1.000000 | -0.317084 | 0.317084 | -0.999588 | -0.000001 | 1001472 |
| component.total_reward | 2.556988 | 2.606224 | 1.000000 | 2.556988 | 2.606224 | -0.956792 | 4.996854 | 1001472 |
| generated_reward | 2.556988 | 2.606224 | 1.000000 | 2.556988 | 2.606224 | -0.956792 | 4.996854 | 1001472 |
| original_env_reward | 0.028696 | 1.258465 | 1.000000 | 0.028696 | 1.258465 | -100.000000 | 132.122254 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -1.630224 | 1.630224 | -8.970000 | -0.310000 | 3531 |
| progress_reward | 1.082942 | 1.082942 | 0.065743 | 1.420196 | 3531 |
| soft_landing_proxy | 814.644893 | 814.644893 | 8.679172 | 4615.391214 | 3531 |
| speed_tracking_reward | -89.916845 | 89.916845 | -836.564245 | -26.060132 | 3531 |
| total_reward | 724.180766 | 727.719140 | -89.206232 | 4562.768920 | 3531 |
