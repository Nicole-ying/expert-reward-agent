# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_proximity | 0.103923 | 0.110823 | 1.000000 | 0.103923 | 0.110823 | -0.156565 | 0.222290 | 1003520 |
| component.landing_progress | 0.751589 | 0.751589 | 0.024133 | 31.143571 | 31.143571 | 0.000000 | 110.000000 | 1003520 |
| component.orientation_penalty | -0.007931 | 0.007931 | 1.000000 | -0.007931 | 0.007931 | -0.713387 | -0.000000 | 1003520 |
| component.total_reward | 0.298715 | 0.322746 | 1.000000 | 0.298715 | 0.322746 | -0.757410 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.034998 | 0.034998 | 0.997297 | -0.035093 | 0.035093 | -0.177851 | -0.000000 | 1003520 |
| generated_reward | 0.298715 | 0.322746 | 1.000000 | 0.298715 | 0.322746 | -0.757410 | 20.000000 | 1003520 |
| original_env_reward | -1.436981 | 2.501953 | 1.000000 | -1.436981 | 2.501953 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_proximity | 7.707139 | 7.707139 | 2.705780 | 8.085408 | 13529 |
| landing_progress | 55.749501 | 55.749501 | 0.000000 | 8190.000000 | 13529 |
| orientation_penalty | -0.588220 | 0.588220 | -10.798781 | -0.013678 | 13529 |
| total_reward | 22.156204 | 22.160787 | -5.916906 | 1756.058037 | 13529 |
| velocity_penalty | -2.595748 | 2.595748 | -4.401342 | -1.626848 | 13529 |
