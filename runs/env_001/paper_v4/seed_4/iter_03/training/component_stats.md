# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.012278 | 0.012278 | 0.613877 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.progress_reward | 0.005462 | 0.005995 | 0.997766 | 0.005475 | 0.006009 | -0.053898 | 0.067501 | 1003520 |
| component.safe_contact_reward | 0.402523 | 0.402523 | 0.619721 | 0.649524 | 0.649524 | 0.000000 | 0.997615 | 1003520 |
| component.stability_penalty | -0.019044 | 0.019044 | 1.000000 | -0.019044 | 0.019044 | -4.971599 | -0.000000 | 1003520 |
| component.total_reward | 0.376664 | 0.414969 | 1.000000 | 0.376664 | 0.414969 | -4.972484 | 0.996504 | 1003520 |
| generated_reward | 0.376664 | 0.414969 | 1.000000 | 0.376664 | 0.414969 | -4.972484 | 0.996504 | 1003520 |
| original_env_reward | 0.035864 | 1.216041 | 1.000000 | 0.035864 | 1.216041 | -100.000000 | 133.841105 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -5.293987 | 5.293987 | -17.260000 | -0.680000 | 2325 |
| progress_reward | 2.355000 | 2.355578 | -0.448384 | 2.842838 | 2325 |
| safe_contact_reward | 173.588729 | 173.588729 | 0.000000 | 787.132029 | 2325 |
| stability_penalty | -8.214752 | 8.214752 | -55.239043 | -2.495826 | 2325 |
| total_reward | 162.434990 | 174.119048 | -54.663970 | 777.315178 | 2325 |
