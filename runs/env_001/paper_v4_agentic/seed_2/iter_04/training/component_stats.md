# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_bonus | 0.009966 | 0.009966 | 0.000716 | 13.909670 | 13.909670 | 0.000000 | 19.420244 | 1003520 |
| component.orientation_penalty | -0.111689 | 0.111689 | 1.000000 | -0.111689 | 0.111689 | -179.111267 | -0.000000 | 1003520 |
| component.proximity_delta | 0.787159 | 0.835045 | 0.999991 | 0.787166 | 0.835053 | -2.046242 | 2.112252 | 1003520 |
| component.total_reward | 0.589572 | 0.779692 | 1.000000 | 0.589572 | 0.779692 | -20.000000 | 19.419222 | 1003520 |
| component.velocity_danger | -0.120129 | 0.120129 | 1.000000 | -0.120129 | 0.120129 | -0.608874 | -0.000000 | 1003520 |
| generated_reward | 0.589572 | 0.779692 | 1.000000 | 0.589572 | 0.779692 | -20.000000 | 19.419222 | 1003520 |
| original_env_reward | -1.502315 | 2.437182 | 1.000000 | -1.502315 | 2.437182 | -100.000000 | 133.705883 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 0.711110 | 0.711110 | 0.000000 | 9854.250875 | 14064 |
| orientation_penalty | -7.969375 | 7.969375 | -2712.457931 | -0.039378 | 14064 |
| proximity_delta | 56.162663 | 56.164241 | -11.090761 | 70.203008 | 14064 |
| total_reward | 42.064649 | 48.844065 | -803.245928 | 9607.405300 | 14064 |
| velocity_danger | -8.571150 | 8.571150 | -15.952490 | -5.040397 | 14064 |
