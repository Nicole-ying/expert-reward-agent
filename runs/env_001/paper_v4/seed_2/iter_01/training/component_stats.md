# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_completion | 0.412478 | 0.412478 | 0.485941 | 0.848822 | 0.848822 | 0.000000 | 0.999999 | 1003520 |
| component.position_proximity | 0.746078 | 0.746078 | 1.000000 | 0.746078 | 0.746078 | 0.095338 | 0.999982 | 1003520 |
| component.soft_landing_velocity | -0.002222 | 0.002222 | 0.999741 | -0.002223 | 0.002223 | -0.797036 | -0.000000 | 1003520 |
| component.stable_orientation | -0.016401 | 0.016401 | 0.999731 | -0.016406 | 0.016406 | -5.345358 | -0.000000 | 1003520 |
| component.total_reward | 1.139932 | 1.142621 | 1.000000 | 1.139932 | 1.142621 | -4.919477 | 1.999902 | 1003520 |
| generated_reward | 1.139932 | 1.142621 | 1.000000 | 1.139932 | 1.142621 | -4.919477 | 1.999902 | 1003520 |
| original_env_reward | -0.033237 | 1.560145 | 1.000000 | -0.033237 | 1.560145 | -100.000000 | 111.046945 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_completion | 241.956169 | 241.956169 | 0.000000 | 843.334335 | 1704 |
| position_proximity | 438.069787 | 438.069787 | 31.626999 | 942.443039 | 1704 |
| soft_landing_velocity | -1.307291 | 1.307291 | -9.312861 | -0.261005 | 1704 |
| stable_orientation | -9.645575 | 9.645575 | -139.021869 | -0.043101 | 1704 |
| total_reward | 669.073090 | 669.341370 | -81.296674 | 1779.068135 | 1704 |
