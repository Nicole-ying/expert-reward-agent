# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_penalty | -0.018455 | 0.018455 | 0.999960 | -0.018456 | 0.018456 | -7.237466 | -0.000000 | 1003520 |
| component.proximity_progress | 0.136671 | 0.150083 | 0.998520 | 0.136873 | 0.150305 | -1.224873 | 1.502343 | 1003520 |
| component.soft_landing_bonus | 1.724621 | 1.724621 | 0.525126 | 3.284207 | 3.284207 | 0.000000 | 3.999852 | 1003520 |
| component.total_reward | 1.791479 | 1.825697 | 1.000000 | 1.791479 | 1.825697 | -9.399718 | 3.999847 | 1003520 |
| component.velocity_penalty | -0.051358 | 0.051358 | 0.999183 | -0.051400 | 0.051400 | -2.358104 | -0.000000 | 1003520 |
| generated_reward | 1.791479 | 1.825697 | 1.000000 | 1.791479 | 1.825697 | -9.399718 | 3.999847 | 1003520 |
| original_env_reward | -0.009224 | 1.422589 | 1.000000 | -0.009224 | 1.422589 | -100.000000 | 125.826464 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -6.363103 | 6.363103 | -106.709472 | -0.010926 | 2908 |
| proximity_progress | 47.091414 | 47.091414 | 12.670982 | 56.563086 | 2908 |
| soft_landing_bonus | 592.596318 | 592.596318 | 0.000000 | 3280.879719 | 2908 |
| total_reward | 615.618512 | 618.631046 | -121.890212 | 3319.444955 | 2908 |
| velocity_penalty | -17.706118 | 17.706118 | -64.135419 | -5.756860 | 2908 |
