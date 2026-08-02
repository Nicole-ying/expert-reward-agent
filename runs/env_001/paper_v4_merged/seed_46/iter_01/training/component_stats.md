# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_proximity_progress | 0.002145 | 0.002377 | 0.999553 | 0.002146 | 0.002378 | -0.021857 | 0.040486 | 1003520 |
| component.landing_gentleness_penalty | -0.001479 | 0.001479 | 0.016430 | -0.089997 | 0.089997 | -0.669432 | -0.000000 | 1003520 |
| component.orientation_penalty | -0.001596 | 0.001596 | 0.046031 | -0.034674 | 0.034674 | -0.626249 | -0.000000 | 1003520 |
| component.terminal_success_bonus | 0.126829 | 0.126829 | 0.634143 | 0.200000 | 0.200000 | 0.000000 | 0.200000 | 1003520 |
| component.total_reward | 0.125899 | 0.131335 | 0.999974 | 0.125902 | 0.131338 | -0.900816 | 0.208491 | 1003520 |
| generated_reward | 0.125899 | 0.131335 | 0.999974 | 0.125902 | 0.131338 | -0.900816 | 0.208491 | 1003520 |
| original_env_reward | 0.057246 | 1.203991 | 1.000000 | 0.057246 | 1.203991 | -100.000000 | 126.229795 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_proximity_progress | 1.184883 | 1.184940 | -0.052271 | 1.418351 | 1812 |
| landing_gentleness_penalty | -0.818916 | 0.818916 | -12.625454 | 0.000000 | 1812 |
| orientation_penalty | -0.883932 | 0.883932 | -20.602151 | 0.000000 | 1812 |
| terminal_success_bonus | 70.074503 | 70.074503 | 0.000000 | 170.000000 | 1812 |
| total_reward | 69.556539 | 71.963664 | -23.287517 | 171.220783 | 1812 |
