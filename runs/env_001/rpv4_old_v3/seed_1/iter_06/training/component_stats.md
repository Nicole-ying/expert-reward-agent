# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_approach | 0.016056 | 0.016995 | 0.999993 | 0.016056 | 0.016995 | -0.040704 | 0.042332 | 1003520 |
| component.landing_comfort | 0.005603 | 0.005603 | 0.006559 | 0.854272 | 0.854272 | 0.000000 | 0.999999 | 1003520 |
| component.soft_landing | -0.280366 | 0.280366 | 1.000000 | -0.280366 | 0.280366 | -2.140822 | -0.000000 | 1003520 |
| component.total_reward | -0.258707 | 0.269488 | 1.000000 | -0.258707 | 0.269488 | -2.156742 | 0.999681 | 1003520 |
| generated_reward | -0.258707 | 0.269488 | 1.000000 | -0.258707 | 0.269488 | -2.156742 | 0.999681 | 1003520 |
| original_env_reward | -1.536714 | 2.403396 | 1.000000 | -1.536714 | 2.403396 | -100.000000 | 141.855010 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_approach | 1.129181 | 1.129181 | 0.001022 | 1.407995 | 14268 |
| landing_comfort | 0.394086 | 0.394086 | 0.000000 | 459.077012 | 14268 |
| soft_landing | -19.717429 | 19.717429 | -85.221112 | -12.695244 | 14268 |
| total_reward | -18.194163 | 18.249320 | -84.827454 | 393.493160 | 14268 |
