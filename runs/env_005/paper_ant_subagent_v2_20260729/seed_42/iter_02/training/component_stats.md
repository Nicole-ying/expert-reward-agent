# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 0.591640 | 0.591640 | 0.498523 | 1.186785 | 1.186785 | 0.000000 | 6.093567 | 1001472 |
| component.lateral_drift_hinge | -0.101538 | 0.101538 | 0.444576 | -0.228392 | 0.228392 | -2.143947 | -0.000000 | 1001472 |
| component.roll_pitch_hinge | -0.085960 | 0.085960 | 0.423265 | -0.203088 | 0.203088 | -3.155825 | -0.000000 | 1001472 |
| component.total_reward | 0.404142 | 0.539164 | 0.684318 | 0.590577 | 0.787886 | -5.063961 | 6.093567 | 1001472 |
| generated_reward | 0.404142 | 0.539164 | 0.684318 | 0.590577 | 0.787886 | -5.063961 | 6.093567 | 1001472 |
| original_env_reward | -0.830111 | 1.245034 | 1.000000 | -0.830111 | 1.245034 | -5.618846 | 4.997473 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 225.544121 | 225.544121 | 0.000000 | 1812.548399 | 2624 |
| lateral_drift_hinge | -38.724104 | 38.724104 | -235.314872 | -0.038326 | 2624 |
| roll_pitch_hinge | -32.784878 | 32.784878 | -214.575059 | -0.312791 | 2624 |
| total_reward | 154.035138 | 172.125708 | -279.693563 | 1539.876952 | 2624 |
