# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.023619 | 0.023619 | 1.000000 | -0.023619 | 0.023619 | -0.050000 | -0.000061 | 1015808 |
| component.angular_velocity_penalty | -0.001136 | 0.001136 | 0.999943 | -0.001136 | 0.001136 | -0.050824 | -0.000000 | 1015808 |
| component.forward_progress | 0.169325 | 0.176262 | 1.000000 | 0.169325 | 0.176262 | -0.378075 | 0.745816 | 1015808 |
| component.tilt_penalty | -0.077915 | 0.077915 | 0.999997 | -0.077916 | 0.077916 | -40.270579 | -0.000000 | 1015808 |
| component.total_reward | 0.067019 | 0.216815 | 1.000000 | 0.067019 | 0.216815 | -20.000000 | 0.732741 | 1015808 |
| generated_reward | 0.067019 | 0.216815 | 1.000000 | 0.067019 | 0.216815 | -20.000000 | 0.732741 | 1015808 |
| original_env_reward | -0.389246 | 0.574039 | 1.000000 | -0.389246 | 0.574039 | -100.000000 | 0.639708 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -5.057228 | 5.057228 | -42.775108 | -0.786975 | 4732 |
| angular_velocity_penalty | -0.243462 | 0.243462 | -2.438050 | -0.043238 | 4732 |
| forward_progress | 36.089934 | 36.551382 | -17.311072 | 495.141168 | 4732 |
| tilt_penalty | -16.719463 | 16.719463 | -1304.066429 | -2.455802 | 4732 |
| total_reward | 14.148059 | 39.065647 | -1350.509865 | 460.829000 | 4732 |
