# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_cost | -0.012867 | 0.012867 | 0.085783 | -0.150000 | 0.150000 | -0.150000 | -0.000000 | 1003520 |
| component.proximity_reward | 0.252882 | 0.269299 | 0.999413 | 0.253031 | 0.269457 | -0.417703 | 0.483956 | 1003520 |
| component.soft_landing_bonus | 0.031786 | 0.031786 | 0.017149 | 1.853545 | 1.853545 | 0.000000 | 1.999983 | 1003520 |
| component.total_reward | 0.212361 | 0.274521 | 0.999945 | 0.212373 | 0.274536 | -0.645884 | 2.000661 | 1003520 |
| component.vertical_descent_penalty | -0.059439 | 0.059439 | 0.736585 | -0.080696 | 0.080696 | -0.430070 | -0.000000 | 1003520 |
| generated_reward | 0.212361 | 0.274521 | 0.999945 | 0.212373 | 0.274536 | -0.645884 | 2.000661 | 1003520 |
| original_env_reward | -1.586335 | 2.297191 | 1.000000 | -1.586335 | 2.297191 | -100.000000 | 125.484778 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_cost | -0.901708 | 0.901708 | -114.600000 | 0.000000 | 14319 |
| proximity_reward | 17.722017 | 17.733081 | -6.819534 | 19.382250 | 14319 |
| soft_landing_bonus | 2.227645 | 2.227645 | 0.000000 | 933.876147 | 14319 |
| total_reward | 14.882265 | 15.090076 | -25.935072 | 825.767309 | 14319 |
| vertical_descent_penalty | -4.165690 | 4.165690 | -6.345778 | -0.097559 | 14319 |
