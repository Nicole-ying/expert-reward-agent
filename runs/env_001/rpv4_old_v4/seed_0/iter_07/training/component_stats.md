# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.009783 | 0.009783 | 0.670755 | -0.014584 | 0.014584 | -0.020000 | 0.000000 | 1003520 |
| component.orientation_penalty | -0.001878 | 0.001878 | 0.999887 | -0.001878 | 0.001878 | -1.532566 | -0.000000 | 1003520 |
| component.progress_reward | 0.011365 | 0.012000 | 0.981028 | 0.011585 | 0.012232 | -0.058768 | 0.076984 | 1003520 |
| component.total_reward | -0.019476 | 0.020290 | 0.999975 | -0.019477 | 0.020290 | -2.077684 | 0.008964 | 1003520 |
| component.vel_penalty | -0.019181 | 0.019181 | 0.983304 | -0.019507 | 0.019507 | -1.275405 | -0.000000 | 1003520 |
| generated_reward | -0.019476 | 0.020290 | 0.999975 | -0.019477 | 0.020290 | -2.077684 | 0.008964 | 1003520 |
| original_env_reward | 0.272449 | 2.451630 | 1.000000 | 0.272449 | 2.451630 | -100.000000 | 112.301566 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -2.100411 | 2.100411 | -8.775000 | -0.315000 | 4671 |
| orientation_penalty | -0.403116 | 0.403116 | -18.408481 | -0.002214 | 4671 |
| progress_reward | 2.440350 | 2.441401 | -1.500475 | 2.839346 | 4671 |
| total_reward | -4.182660 | 4.182660 | -32.567262 | -1.738511 | 4671 |
| vel_penalty | -4.119483 | 4.119483 | -20.213987 | -2.037014 | 4671 |
