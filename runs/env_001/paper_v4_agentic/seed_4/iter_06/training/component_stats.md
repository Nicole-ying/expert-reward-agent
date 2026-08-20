# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.136033 | 0.136033 | 0.680165 | -0.200000 | 0.200000 | -0.200000 | -0.000000 | 1003520 |
| component.landing_prep | 1.496287 | 1.496287 | 1.000000 | 1.496287 | 1.496287 | 0.000381 | 2.997228 | 1003520 |
| component.progress_gated | 0.312502 | 0.312502 | 0.642654 | 0.486267 | 0.486267 | 0.000000 | 1.998967 | 1003520 |
| component.total_reward | 1.672755 | 1.684910 | 1.000000 | 1.672755 | 1.684910 | -0.199359 | 3.514246 | 1003520 |
| generated_reward | 1.672755 | 1.684910 | 1.000000 | 1.672755 | 1.684910 | -0.199359 | 3.514246 | 1003520 |
| original_env_reward | -0.021049 | 1.413306 | 1.000000 | -0.021049 | 1.413306 | -100.000000 | 171.280599 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -57.375558 | 57.375558 | -170.800000 | -2.400000 | 2373 |
| landing_prep | 631.022796 | 631.022796 | 0.151474 | 2446.611389 | 2373 |
| progress_gated | 131.772689 | 131.772689 | 0.000000 | 397.616995 | 2373 |
| total_reward | 705.419919 | 706.138275 | -19.774074 | 2571.826184 | 2373 |
