# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_distance_progress | 0.011828 | 0.014567 | 0.999380 | 0.011836 | 0.014576 | -0.178311 | 0.170965 | 1003520 |
| component.landing_contact_success | 1.333343 | 1.333343 | 0.496483 | 2.685574 | 2.685574 | 0.000000 | 4.983787 | 1003520 |
| component.orientation_penalty | -0.015742 | 0.015742 | 0.999811 | -0.015745 | 0.015745 | -7.678077 | -0.000000 | 1003520 |
| component.soft_landing | -0.037626 | 0.037626 | 0.999850 | -0.037632 | 0.037632 | -8.347300 | -0.000000 | 1003520 |
| component.total_reward | 1.291802 | 1.372364 | 1.000000 | 1.291802 | 1.372364 | -10.113285 | 4.983787 | 1003520 |
| generated_reward | 1.291802 | 1.372364 | 1.000000 | 1.291802 | 1.372364 | -10.113285 | 4.983787 | 1003520 |
| original_env_reward | -0.098945 | 1.482808 | 1.000000 | -0.098945 | 1.482808 | -100.000000 | 128.018975 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_distance_progress | 4.757775 | 5.036048 | -39.490711 | 7.077874 | 2490 |
| landing_contact_success | 536.480096 | 536.480096 | 0.000000 | 4028.281362 | 2490 |
| orientation_penalty | -6.340343 | 6.340343 | -186.776592 | -0.006132 | 2490 |
| soft_landing | -15.146440 | 15.146440 | -125.026462 | -1.696851 | 2490 |
| total_reward | 519.751088 | 544.209183 | -201.857728 | 4008.988405 | 2490 |
