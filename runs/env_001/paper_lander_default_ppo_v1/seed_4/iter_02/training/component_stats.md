# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 0.275721 | 0.275721 | 0.605584 | 0.455298 | 0.455298 | 0.000000 | 0.499979 | 1001472 |
| component.orientation_penalty | -0.003911 | 0.003911 | 0.999775 | -0.003912 | 0.003912 | -4.623415 | -0.000000 | 1001472 |
| component.proximity_reward | 0.003861 | 0.005033 | 0.999931 | 0.003862 | 0.005034 | -0.093332 | 0.075328 | 1001472 |
| component.speed_penalty_gated | -0.012270 | 0.012270 | 0.999983 | -0.012270 | 0.012270 | -1.373294 | -0.000000 | 1001472 |
| component.total_reward | 0.263401 | 0.286621 | 1.000000 | 0.263401 | 0.286621 | -4.955027 | 0.499953 | 1001472 |
| generated_reward | 0.263401 | 0.286621 | 1.000000 | 0.263401 | 0.286621 | -4.955027 | 0.499953 | 1001472 |
| original_env_reward | 0.036333 | 1.132181 | 1.000000 | 0.036333 | 1.132181 | -100.000000 | 123.955924 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 151.801487 | 151.801487 | 0.000000 | 439.591688 | 1819 |
| orientation_penalty | -2.152166 | 2.152166 | -39.590089 | -0.049098 | 1819 |
| proximity_reward | 2.124384 | 2.471073 | -14.583905 | 2.841626 | 1819 |
| speed_penalty_gated | -6.750336 | 6.750336 | -34.925702 | -1.728706 | 1819 |
| total_reward | 145.023368 | 153.962878 | -66.260468 | 432.985626 | 1819 |
