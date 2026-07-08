# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_delta | 0.011031 | 0.012943 | 0.997795 | 0.011055 | 0.012972 | -0.100357 | 0.116362 | 1003520 |
| component.soft_landing_reward | 0.570248 | 0.570248 | 1.000000 | 0.570248 | 0.570248 | 0.000000 | 1.000000 | 1003520 |
| component.stability_penalty | -0.025764 | 0.025764 | 1.000000 | -0.025764 | 0.025764 | -0.485676 | -0.000001 | 1003520 |
| component.total_reward | 0.555515 | 0.565471 | 1.000000 | 0.555515 | 0.565471 | -0.495365 | 0.999986 | 1003520 |
| generated_reward | 0.555515 | 0.565471 | 1.000000 | 0.555515 | 0.565471 | -0.495365 | 0.999986 | 1003520 |
| original_env_reward | -0.181004 | 2.390369 | 1.000000 | -0.181004 | 2.390369 | -100.000000 | 130.372096 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_delta | 3.527073 | 3.527073 | 0.767706 | 4.259464 | 3134 |
| soft_landing_reward | 182.131439 | 182.131439 | 0.000004 | 894.693525 | 3134 |
| stability_penalty | -8.241855 | 8.241855 | -38.909414 | -4.207887 | 3134 |
| total_reward | 177.416657 | 179.650284 | -12.907216 | 891.947127 | 3134 |
