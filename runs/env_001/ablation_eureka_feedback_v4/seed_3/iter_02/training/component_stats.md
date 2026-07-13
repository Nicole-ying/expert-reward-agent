# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.008090 | 0.008090 | 0.809023 | -0.010000 | 0.010000 | -0.010000 | 0.000000 | 1003520 |
| component.proximity | 0.679157 | 0.679157 | 1.000000 | 0.679157 | 0.679157 | 0.097801 | 0.999990 | 1003520 |
| component.soft_landing | 0.760758 | 0.760758 | 0.999957 | 0.760791 | 0.760791 | 0.000000 | 1.443609 | 1003520 |
| component.terminal_velocity_penalty | -0.000081 | 0.000081 | 0.000783 | -0.103241 | 0.103241 | -0.759388 | 0.000000 | 1003520 |
| component.total_reward | 1.431745 | 1.431745 | 1.000000 | 1.431745 | 1.431745 | 0.111605 | 2.000140 | 1003520 |
| generated_reward | 1.431745 | 1.431745 | 1.000000 | 1.431745 | 1.431745 | 0.111605 | 2.000140 | 1003520 |
| original_env_reward | -0.097122 | 3.096545 | 1.000000 | -0.097122 | 3.096545 | -100.000000 | 135.593962 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -4.323054 | 4.323054 | -9.080000 | -0.370000 | 1876 |
| proximity | 362.903148 | 362.903148 | 15.754175 | 926.806284 | 1876 |
| soft_landing | 406.553915 | 406.553915 | 2.528331 | 1081.530026 | 1876 |
| terminal_velocity_penalty | -0.043255 | 0.043255 | -1.720502 | 0.000000 | 1876 |
| total_reward | 765.090758 | 765.090758 | 19.002941 | 1841.941855 | 1876 |
