# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_proximity | 0.012485 | 0.014556 | 0.999997 | 0.012485 | 0.014556 | -0.120043 | 0.167186 | 1003520 |
| component.landing_approach | 0.066978 | 0.066978 | 0.613506 | 0.109172 | 0.109172 | 0.000000 | 18.969340 | 1003520 |
| component.total_reward | -0.038531 | 0.134026 | 1.000000 | -0.038531 | 0.134026 | -20.000000 | 18.913447 | 1003520 |
| component.upright_penalty | -0.061265 | 0.061265 | 1.000000 | -0.061265 | 0.061265 | -58.809209 | -0.000000 | 1003520 |
| component.velocity_penalty | -0.058352 | 0.058352 | 0.999855 | -0.058361 | 0.058361 | -3.435310 | -0.000000 | 1003520 |
| generated_reward | -0.038531 | 0.134026 | 1.000000 | -0.038531 | 0.134026 | -20.000000 | 18.913447 | 1003520 |
| original_env_reward | -0.135700 | 3.403725 | 1.000000 | -0.135700 | 3.403725 | -100.000000 | 124.805484 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_proximity | 4.636609 | 4.637553 | -1.132797 | 5.681293 | 2698 |
| landing_approach | 24.862709 | 24.862709 | 0.000000 | 98.893147 | 2698 |
| total_reward | -14.345504 | 36.654561 | -393.582417 | 52.946381 | 2698 |
| upright_penalty | -22.762197 | 22.762197 | -397.595412 | -0.130758 | 2698 |
| velocity_penalty | -21.686444 | 21.686444 | -69.737697 | -6.005726 | 2698 |
