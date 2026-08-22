# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 1.921371 | 1.921371 | 0.014416 | 133.278081 | 133.278081 | 0.000000 | 202.000000 | 1003520 |
| component.goal_proximity | -0.048511 | 0.048511 | 1.000000 | -0.048511 | 0.048511 | -0.085483 | -0.000035 | 1003520 |
| component.orientation_penalty | -0.005249 | 0.005249 | 1.000000 | -0.005249 | 0.005249 | -0.358664 | -0.000000 | 1003520 |
| component.total_reward | 0.038733 | 0.359150 | 1.000000 | 0.038733 | 0.359150 | -0.646405 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.109695 | 0.109695 | 0.997419 | -0.109979 | 0.109979 | -0.529869 | -0.000000 | 1003520 |
| generated_reward | 0.038733 | 0.359150 | 1.000000 | 0.038733 | 0.359150 | -0.646405 | 20.000000 | 1003520 |
| original_env_reward | -1.455627 | 2.601745 | 1.000000 | -1.455627 | 2.601745 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 138.396067 | 138.396067 | 0.000000 | 404.000000 | 13932 |
| goal_proximity | -3.494005 | 3.494005 | -30.757568 | -2.036105 | 13932 |
| orientation_penalty | -0.378083 | 0.378083 | -6.119912 | -0.011967 | 13932 |
| total_reward | 2.790271 | 10.738323 | -44.289573 | 30.264006 | 13932 |
| velocity_penalty | -7.901207 | 7.901207 | -14.961014 | -5.223991 | 13932 |
