# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_delta | 0.048372 | 0.051146 | 0.999996 | 0.048373 | 0.051146 | -0.117043 | 0.126612 | 1003520 |
| component.soft_landing_reward | 0.004556 | 0.004556 | 0.923782 | 0.004932 | 0.004932 | 0.000000 | 0.978450 | 1003520 |
| component.stability_penalty | -0.111332 | 0.111332 | 1.000000 | -0.111332 | 0.111332 | -0.563551 | -0.000000 | 1003520 |
| component.total_reward | -0.058404 | 0.066729 | 1.000000 | -0.058404 | 0.066729 | -0.627722 | 0.978908 | 1003520 |
| generated_reward | -0.058404 | 0.066729 | 1.000000 | -0.058404 | 0.066729 | -0.627722 | 0.978908 | 1003520 |
| original_env_reward | -1.570053 | 2.423711 | 1.000000 | -1.570053 | 2.423711 | -100.000000 | 126.782621 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_delta | 3.391580 | 3.391580 | 0.989073 | 4.243407 | 14312 |
| soft_landing_reward | 0.319434 | 0.319434 | 0.000000 | 16.688668 | 14312 |
| stability_penalty | -7.805822 | 7.805822 | -19.155082 | -6.303836 | 14312 |
| total_reward | -4.094808 | 4.099449 | -17.666499 | 11.222666 | 14312 |
