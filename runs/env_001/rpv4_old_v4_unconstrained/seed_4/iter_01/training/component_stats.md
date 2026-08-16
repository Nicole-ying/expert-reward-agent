# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.attitude_constraint | -0.020761 | 0.023877 | 1.000000 | -0.020761 | 0.023877 | -4.275114 | 2.740158 | 1003520 |
| component.fuel_efficiency | -0.006014 | 0.006014 | 0.120274 | -0.050000 | 0.050000 | -0.050000 | -0.000000 | 1003520 |
| component.goal_proximity | 0.030607 | 0.035558 | 0.999995 | 0.030607 | 0.035558 | -0.116830 | 0.089588 | 1003520 |
| component.landing_contact | 0.010641 | 0.010641 | 0.003304 | 3.220145 | 3.220145 | 0.000000 | 4.000000 | 1003520 |
| component.settling_stability | 0.006556 | 0.042683 | 1.000000 | 0.006556 | 0.042683 | -0.196237 | 3.671339 | 1003520 |
| component.total_reward | 0.021028 | 0.061224 | 1.000000 | 0.021028 | 0.061224 | -3.627361 | 7.658848 | 1003520 |
| generated_reward | 0.021028 | 0.061224 | 1.000000 | 0.021028 | 0.061224 | -3.627361 | 7.658848 | 1003520 |
| original_env_reward | -1.746240 | 2.588061 | 1.000000 | -1.746240 | 2.588061 | -100.000000 | 146.040420 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_constraint | -1.466580 | 1.505992 | -4.328494 | 0.838342 | 14206 |
| fuel_efficiency | -0.424747 | 0.424747 | -33.900000 | 0.000000 | 14206 |
| goal_proximity | 2.162003 | 2.162306 | -1.763010 | 2.813150 | 14206 |
| landing_contact | 0.751654 | 0.751654 | 0.000000 | 4.000000 | 14206 |
| settling_stability | 0.463200 | 0.751727 | -2.130226 | 2.453104 | 14206 |
| total_reward | 1.485531 | 2.851394 | -31.169736 | 9.107019 | 14206 |
