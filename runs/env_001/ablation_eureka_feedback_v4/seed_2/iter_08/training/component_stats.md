# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_cost | -0.012921 | 0.012921 | 0.086137 | -0.150000 | 0.150000 | -0.150000 | -0.000000 | 1003520 |
| component.proximity_reward | 0.254689 | 0.270878 | 0.999488 | 0.254819 | 0.271017 | -0.387958 | 0.483428 | 1003520 |
| component.soft_landing_bonus | 0.022741 | 0.022741 | 0.017481 | 1.300857 | 1.300857 | 0.000000 | 1.998944 | 1003520 |
| component.total_reward | 0.264496 | 0.289190 | 0.999899 | 0.264522 | 0.289219 | -0.493366 | 1.998949 | 1003520 |
| component.velocity_penalty | -0.000013 | 0.000013 | 0.002166 | -0.006185 | 0.006185 | -0.128601 | -0.000000 | 1003520 |
| generated_reward | 0.264496 | 0.289190 | 0.999899 | 0.264522 | 0.289219 | -0.493366 | 1.998949 | 1003520 |
| original_env_reward | -1.589297 | 2.291252 | 1.000000 | -1.589297 | 2.291252 | -100.000000 | 118.466224 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_cost | -0.902977 | 0.902977 | -114.600000 | 0.000000 | 14358 |
| proximity_reward | 17.798885 | 17.807744 | -7.636586 | 19.384312 | 14358 |
| soft_landing_bonus | 1.589422 | 1.589422 | 0.000000 | 889.806847 | 14358 |
| total_reward | 18.484393 | 18.602221 | -22.319338 | 785.144047 | 14358 |
| velocity_penalty | -0.000936 | 0.000936 | -1.032751 | 0.000000 | 14358 |
