# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.009327 | 0.009327 | 0.999931 | -0.009327 | 0.009327 | -7.655549 | -0.000000 | 1003520 |
| component.approach_progress | 0.005212 | 0.005703 | 0.999651 | 0.005214 | 0.005705 | -0.068628 | 0.077302 | 1003520 |
| component.landing_quality | 0.073504 | 0.073504 | 1.000000 | 0.073504 | 0.073504 | 0.012566 | 0.099978 | 1003520 |
| component.total_reward | 0.021596 | 0.113569 | 1.000000 | 0.021596 | 0.113569 | -8.085008 | 0.099973 | 1003520 |
| component.velocity_penalty | -0.047794 | 0.047794 | 0.272376 | -0.175470 | 0.175470 | -1.255888 | -0.000000 | 1003520 |
| generated_reward | 0.021596 | 0.113569 | 1.000000 | 0.021596 | 0.113569 | -8.085008 | 0.099973 | 1003520 |
| original_env_reward | -0.030531 | 1.785804 | 1.000000 | -0.030531 | 1.785804 | -100.000000 | 123.631264 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -4.212468 | 4.212468 | -126.065422 | -0.021034 | 2220 |
| approach_progress | 2.351394 | 2.357062 | -2.203485 | 2.839618 | 2220 |
| landing_quality | 33.167129 | 33.167129 | 1.370685 | 89.088496 | 2220 |
| total_reward | 9.712076 | 46.862305 | -170.231472 | 85.033545 | 2220 |
| velocity_penalty | -21.593978 | 21.593978 | -99.030637 | -0.547067 | 2220 |
