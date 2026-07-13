# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_completion | 0.732726 | 0.732726 | 1.000000 | 0.732726 | 0.732726 | 0.077510 | 1.000000 | 1003520 |
| component.position_proximity | 0.716494 | 0.716494 | 1.000000 | 0.716494 | 0.716494 | 0.053651 | 0.999978 | 1003520 |
| component.soft_landing_velocity | -0.003064 | 0.003064 | 0.999738 | -0.003064 | 0.003064 | -1.554870 | -0.000000 | 1003520 |
| component.stable_orientation | -0.024302 | 0.024302 | 0.999935 | -0.024304 | 0.024304 | -9.463927 | -0.000000 | 1003520 |
| component.total_reward | 1.421854 | 1.426526 | 1.000000 | 1.421854 | 1.426526 | -8.818251 | 1.999913 | 1003520 |
| generated_reward | 1.421854 | 1.426526 | 1.000000 | 1.421854 | 1.426526 | -8.818251 | 1.999913 | 1003520 |
| original_env_reward | -0.122504 | 2.585077 | 1.000000 | -0.122504 | 2.585077 | -100.000000 | 126.823230 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_completion | 394.355951 | 394.355951 | 24.685533 | 911.497087 | 1861 |
| position_proximity | 385.615997 | 385.615997 | 32.075061 | 926.222443 | 1861 |
| soft_landing_velocity | -1.651196 | 1.651196 | -14.173615 | -0.242311 | 1861 |
| stable_orientation | -13.100013 | 13.100013 | -308.064339 | -0.009319 | 1861 |
| total_reward | 765.220739 | 765.402365 | -83.887165 | 1823.429368 | 1861 |
