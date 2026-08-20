# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_velocity_reward | 0.317853 | 0.595327 | 1.000000 | 0.317853 | 0.595327 | -4.652649 | 4.982860 | 1003520 |
| component.orientation_reward | 0.042593 | 0.472021 | 1.000000 | 0.042593 | 0.472021 | -0.500000 | 0.500000 | 1003520 |
| component.stability_penalty | -0.286487 | 0.286487 | 1.000000 | -0.286487 | 0.286487 | -14.126712 | -0.000053 | 1003520 |
| component.total_reward | 0.073959 | 0.821460 | 1.000000 | 0.073959 | 0.821460 | -15.189050 | 5.229789 | 1003520 |
| generated_reward | 0.073959 | 0.821460 | 1.000000 | 0.073959 | 0.821460 | -15.189050 | 5.229789 | 1003520 |
| original_env_reward | -0.826990 | 1.059719 | 1.000000 | -0.826990 | 1.059719 | -6.160585 | 4.437523 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_velocity_reward | 51.854662 | 55.979301 | -95.856908 | 918.610984 | 6147 |
| orientation_reward | 6.996468 | 70.044508 | -458.957505 | 483.035270 | 6147 |
| stability_penalty | -46.732042 | 46.732042 | -346.882235 | -3.717695 | 6147 |
| total_reward | 12.119088 | 95.050573 | -708.403693 | 1077.650460 | 6147 |
