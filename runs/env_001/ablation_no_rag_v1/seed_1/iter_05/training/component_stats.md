# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.006640 | 0.007267 | 0.997102 | 0.006659 | 0.007288 | -0.037875 | 0.040691 | 1003520 |
| component.landing_quality | 0.655595 | 0.655595 | 0.350736 | 1.869196 | 1.869196 | 0.000000 | 2.000000 | 1003520 |
| component.tilt_penalty | -0.006319 | 0.006319 | 1.000000 | -0.006319 | 0.006319 | -0.341967 | -0.000000 | 1003520 |
| component.total_reward | 0.655916 | 0.660159 | 1.000000 | 0.655916 | 0.660159 | -0.352038 | 2.003932 | 1003520 |
| generated_reward | 0.655916 | 0.660159 | 1.000000 | 0.655916 | 0.660159 | -0.352038 | 2.003932 | 1003520 |
| original_env_reward | -0.385943 | 1.855997 | 1.000000 | -0.385943 | 1.855997 | -100.000000 | 129.557552 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 1.084546 | 1.084546 | 0.016892 | 1.418949 | 6139 |
| landing_quality | 106.514252 | 106.514252 | 0.000000 | 1776.257401 | 6139 |
| tilt_penalty | -1.031151 | 1.031151 | -40.846342 | -0.015076 | 6139 |
| total_reward | 106.567647 | 106.764468 | -9.766522 | 1775.073595 | 6139 |
