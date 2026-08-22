# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 7.374841 | 7.374841 | 0.294994 | 25.000000 | 25.000000 | 0.000000 | 25.000000 | 1003520 |
| component.goal_proximity | -0.029813 | 0.029813 | 1.000000 | -0.029813 | 0.029813 | -0.084661 | -0.000005 | 1003520 |
| component.orientation_penalty | -0.006596 | 0.006596 | 1.000000 | -0.006596 | 0.006596 | -0.326467 | -0.000000 | 1003520 |
| component.total_reward | 5.799542 | 6.000203 | 1.000000 | 5.799542 | 6.000203 | -2.768751 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.069428 | 0.069428 | 0.999506 | -0.069462 | 0.069462 | -2.728768 | -0.000000 | 1003520 |
| generated_reward | 5.799542 | 6.000203 | 1.000000 | 5.799542 | 6.000203 | -2.768751 | 20.000000 | 1003520 |
| original_env_reward | -0.150572 | 2.145899 | 1.000000 | -0.150572 | 2.145899 | -100.000000 | 129.205613 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 1593.678471 | 1593.678471 | 0.000000 | 21175.000000 | 4631 |
| goal_proximity | -6.452538 | 6.452538 | -42.216790 | -2.096311 | 4631 |
| orientation_penalty | -1.426590 | 1.426590 | -20.716086 | -0.023868 | 4631 |
| total_reward | 1253.223771 | 1268.789307 | -128.967760 | 16924.811262 | 4631 |
| velocity_penalty | -15.029959 | 15.029959 | -112.285835 | -4.953167 | 4631 |
