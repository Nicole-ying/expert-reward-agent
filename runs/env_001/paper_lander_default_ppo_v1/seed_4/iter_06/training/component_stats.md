# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.004988 | 0.004988 | 0.999996 | -0.004988 | 0.004988 | -4.417630 | -0.000000 | 1001472 |
| component.proximity_reward | 0.006611 | 0.007803 | 0.999980 | 0.006612 | 0.007803 | -0.061420 | 0.073502 | 1001472 |
| component.settlement_bonus | 15.402570 | 15.402570 | 0.270128 | 57.019447 | 57.019447 | 0.000000 | 396.270314 | 1001472 |
| component.speed_penalty_gated | -0.022855 | 0.022855 | 0.999990 | -0.022856 | 0.022856 | -1.297075 | -0.000000 | 1001472 |
| component.total_reward | 3.991924 | 4.028051 | 1.000000 | 3.991924 | 4.028051 | -4.121120 | 20.000000 | 1001472 |
| generated_reward | 3.991924 | 4.028051 | 1.000000 | 3.991924 | 4.028051 | -4.121120 | 20.000000 | 1001472 |
| original_env_reward | -0.050060 | 3.856315 | 1.000000 | -0.050060 | 3.856315 | -100.000000 | 130.661834 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -1.736377 | 1.736377 | -34.124713 | -0.014629 | 2876 |
| proximity_reward | 2.301304 | 2.305499 | -1.667248 | 2.831774 | 2876 |
| settlement_bonus | 5356.322588 | 5356.322588 | 0.000000 | 32070.906446 | 2876 |
| speed_penalty_gated | -7.956346 | 7.956346 | -21.269280 | -2.661958 | 2876 |
| total_reward | 1388.383112 | 1391.437988 | -36.631371 | 6721.459394 | 2876 |
