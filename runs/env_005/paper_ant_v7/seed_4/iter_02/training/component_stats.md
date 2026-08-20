# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_upright_product | 0.447467 | 0.570096 | 0.584314 | 0.765799 | 0.975667 | -3.890313 | 5.547017 | 1001472 |
| component.height_survival_penalty | -0.017226 | 0.017226 | 0.428978 | -0.040156 | 0.040156 | -0.074997 | -0.000000 | 1001472 |
| component.torque_penalty | -0.016356 | 0.016356 | 1.000000 | -0.016356 | 0.016356 | -0.030000 | -0.000402 | 1001472 |
| component.total_reward | 0.413884 | 0.586709 | 1.000000 | 0.413884 | 0.586709 | -3.907160 | 5.531453 | 1001472 |
| generated_reward | 0.413884 | 0.586709 | 1.000000 | 0.413884 | 0.586709 | -3.907160 | 5.531453 | 1001472 |
| original_env_reward | -0.710655 | 1.045034 | 1.000000 | -0.710655 | 1.045034 | -5.735285 | 4.948805 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_upright_product | 91.485398 | 93.960075 | -163.512168 | 1449.538760 | 4891 |
| height_survival_penalty | -3.527041 | 3.527041 | -43.382549 | 0.000000 | 4891 |
| torque_penalty | -3.347260 | 3.347260 | -17.744816 | -0.095058 | 4891 |
| total_reward | 84.611098 | 90.823407 | -180.674582 | 1431.747131 | 4891 |
