# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_cost | -0.014399 | 0.014399 | 0.095991 | -0.150000 | 0.150000 | -0.150000 | -0.000000 | 1003520 |
| component.proximity_reward | 0.253072 | 0.269603 | 0.999329 | 0.253241 | 0.269784 | -0.402073 | 0.483594 | 1003520 |
| component.soft_landing_bonus | 0.032806 | 0.032806 | 0.017749 | 1.848400 | 1.848400 | 0.000000 | 1.999974 | 1003520 |
| component.total_reward | 0.261097 | 0.290440 | 0.999993 | 0.261099 | 0.290442 | -0.512039 | 2.006353 | 1003520 |
| component.velocity_penalty | -0.010382 | 0.010382 | 0.996430 | -0.010420 | 0.010420 | -0.134034 | -0.000000 | 1003520 |
| generated_reward | 0.261097 | 0.290440 | 0.999993 | 0.261099 | 0.290442 | -0.512039 | 2.006353 | 1003520 |
| original_env_reward | -1.578077 | 2.302176 | 1.000000 | -1.578077 | 2.302176 | -100.000000 | 131.645648 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_cost | -1.009644 | 1.009644 | -114.600000 | 0.000000 | 14310 |
| proximity_reward | 17.744708 | 17.755446 | -5.484541 | 19.384561 | 14310 |
| soft_landing_bonus | 2.300619 | 2.300619 | 0.000000 | 933.903626 | 14310 |
| total_reward | 18.307666 | 18.466033 | -21.534986 | 828.587862 | 14310 |
| velocity_penalty | -0.728017 | 0.728017 | -1.607850 | -0.425705 | 14310 |
