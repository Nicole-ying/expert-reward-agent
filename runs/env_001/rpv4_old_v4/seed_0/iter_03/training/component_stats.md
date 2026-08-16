# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.landing_proximity_reward | 0.013438 | 0.013438 | 0.497643 | 0.027004 | 0.027004 | 0.000000 | 0.048245 | 1003520 |
| component.orientation_penalty | -0.006308 | 0.006308 | 1.000000 | -0.006308 | 0.006308 | -3.784572 | -0.000000 | 1003520 |
| component.progress_reward | -0.001489 | 0.009462 | 1.000000 | -0.001489 | 0.009462 | -0.117804 | 0.049556 | 1003520 |
| component.total_reward | 0.000873 | 0.031943 | 1.000000 | 0.000873 | 0.031943 | -3.957418 | 0.065550 | 1003520 |
| component.vel_damping | -0.004768 | 0.004768 | 0.066502 | -0.071700 | 0.071700 | -1.648216 | -0.000000 | 1003520 |
| generated_reward | 0.000873 | 0.031943 | 1.000000 | 0.000873 | 0.031943 | -3.957418 | 0.065550 | 1003520 |
| original_env_reward | -0.734753 | 2.342430 | 1.000000 | -0.734753 | 2.342430 | -100.000000 | 121.580685 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_proximity_reward | 5.639931 | 5.639931 | 0.000000 | 28.102510 | 2386 |
| orientation_penalty | -2.652624 | 2.652624 | -77.095469 | -0.028696 | 2386 |
| progress_reward | -0.622531 | 0.957982 | -33.310195 | 1.412644 | 2386 |
| total_reward | 0.359337 | 8.743646 | -95.396938 | 27.761448 | 2386 |
| vel_damping | -2.005440 | 2.005440 | -51.134991 | 0.000000 | 2386 |
