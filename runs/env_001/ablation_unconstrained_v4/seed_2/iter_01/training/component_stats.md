# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_establishment | 0.050210 | 0.050210 | 0.033568 | 1.495785 | 1.495785 | 0.000000 | 2.000000 | 1003520 |
| component.goal_proximity | -1.075458 | 1.075458 | 1.000000 | -1.075458 | 1.075458 | -2.364237 | -0.000179 | 1003520 |
| component.soft_landing_stabilization | -0.133279 | 0.133279 | 0.997072 | -0.133670 | 0.133670 | -0.693853 | -0.000000 | 1003520 |
| component.total_reward | -1.199669 | 1.249015 | 1.000000 | -1.199669 | 1.249015 | -20.000000 | 1.985486 | 1003520 |
| component.upright_attitude | -0.041167 | 0.041167 | 0.999997 | -0.041167 | 0.041167 | -24.649101 | -0.000000 | 1003520 |
| generated_reward | -1.199669 | 1.249015 | 1.000000 | -1.199669 | 1.249015 | -20.000000 | 1.985486 | 1003520 |
| original_env_reward | -1.607500 | 2.453006 | 1.000000 | -1.607500 | 2.453006 | -100.000000 | 141.859952 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_establishment | 3.541895 | 3.541895 | 0.000000 | 1117.000000 | 14226 |
| goal_proximity | -75.850188 | 75.850188 | -929.632331 | -40.688750 | 14226 |
| soft_landing_stabilization | -9.401006 | 9.401006 | -19.767488 | -5.017119 | 14226 |
| total_reward | -84.611509 | 84.752781 | -672.029577 | 956.563942 | 14226 |
| upright_attitude | -2.903942 | 2.903942 | -517.016956 | -0.002352 | 14226 |
