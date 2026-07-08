# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.023331 | 0.025509 | 0.999926 | 0.023333 | 0.025511 | -0.074703 | 0.080127 | 1003520 |
| component.landing_guidance | 0.174544 | 0.218692 | 0.999968 | 0.174550 | 0.218699 | -0.141458 | 2.000000 | 1003520 |
| component.tilt_penalty | -0.004525 | 0.004525 | 1.000000 | -0.004525 | 0.004525 | -0.373126 | -0.000000 | 1003520 |
| component.total_reward | 0.193350 | 0.208301 | 1.000000 | 0.193350 | 0.208301 | -0.372313 | 2.054916 | 1003520 |
| generated_reward | 0.193350 | 0.208301 | 1.000000 | 0.193350 | 0.208301 | -0.372313 | 2.054916 | 1003520 |
| original_env_reward | -0.866979 | 2.673349 | 1.000000 | -0.866979 | 2.673349 | -100.000000 | 133.929667 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 2.141343 | 2.141343 | 0.582523 | 2.788228 | 10930 |
| landing_guidance | 15.768075 | 16.723414 | -3.522713 | 1694.036852 | 10930 |
| tilt_penalty | -0.414422 | 0.414422 | -34.701750 | -0.017426 | 10930 |
| total_reward | 17.494996 | 17.740714 | -10.374089 | 1688.758216 | 10930 |
