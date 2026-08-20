# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_approach | 0.015945 | 0.016876 | 0.999999 | 0.015945 | 0.016876 | -0.040295 | 0.042477 | 1003520 |
| component.landing_success | 0.642837 | 0.642837 | 0.006428 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.soft_landing | -0.287575 | 0.287575 | 1.000000 | -0.287575 | 0.287575 | -2.140822 | -0.000000 | 1003520 |
| component.total_reward | -0.142716 | 0.399851 | 1.000000 | -0.142716 | 0.399851 | -2.156742 | 20.000000 | 1003520 |
| generated_reward | -0.142716 | 0.399851 | 1.000000 | -0.142716 | 0.399851 | -2.156742 | 20.000000 | 1003520 |
| original_env_reward | -1.556223 | 2.448453 | 1.000000 | -1.556223 | 2.448453 | -100.000000 | 111.401999 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_approach | 1.128681 | 1.128852 | -0.391908 | 1.393739 | 14176 |
| landing_success | 45.506490 | 45.506490 | 0.000000 | 36900.000000 | 14176 |
| soft_landing | -20.355368 | 20.355368 | -85.221112 | -12.987651 | 14176 |
| total_reward | -10.100805 | 14.545755 | -84.827454 | 7330.478336 | 14176 |
