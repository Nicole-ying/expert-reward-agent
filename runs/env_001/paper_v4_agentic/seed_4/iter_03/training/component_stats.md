# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.017388 | 0.017388 | 0.086939 | -0.200000 | 0.200000 | -0.200000 | -0.000000 | 1003520 |
| component.progress_gated | 0.002181 | 0.002181 | 0.904346 | 0.002412 | 0.002412 | 0.000000 | 0.005940 | 1003520 |
| component.proximity_stability | 0.158488 | 0.158488 | 0.172071 | 0.921063 | 0.921063 | 0.000000 | 13.717318 | 1003520 |
| component.total_reward | 0.143282 | 0.173534 | 0.941242 | 0.152226 | 0.184367 | -0.200000 | 13.717318 | 1003520 |
| generated_reward | 0.143282 | 0.173534 | 0.941242 | 0.152226 | 0.184367 | -0.200000 | 13.717318 | 1003520 |
| original_env_reward | -1.635328 | 2.339535 | 1.000000 | -1.635328 | 2.339535 | -100.000000 | 148.749014 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -1.216426 | 1.216426 | -21.200000 | 0.000000 | 14343 |
| progress_gated | 0.152549 | 0.152549 | 0.000193 | 0.279215 | 14343 |
| proximity_stability | 11.087865 | 11.087865 | 0.000000 | 32.706154 | 14343 |
| total_reward | 10.023987 | 10.773661 | -19.779984 | 30.629329 | 14343 |
