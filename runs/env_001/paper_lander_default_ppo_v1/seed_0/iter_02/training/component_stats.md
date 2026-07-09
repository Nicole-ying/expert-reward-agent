# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_reward | 0.017052 | 0.017804 | 0.999999 | 0.017052 | 0.017804 | -0.038917 | 0.056823 | 1001472 |
| component.soft_landing_proxy | 0.002017 | 0.002017 | 0.004033 | 0.500000 | 0.500000 | 0.000000 | 0.500000 | 1001472 |
| component.stability_penalty | -0.001600 | 0.001600 | 1.000000 | -0.001600 | 0.001600 | -0.006963 | -0.000000 | 1001472 |
| component.total_reward | 0.017468 | 0.018320 | 1.000000 | 0.017468 | 0.018320 | -0.040916 | 0.502249 | 1001472 |
| generated_reward | 0.017468 | 0.018320 | 1.000000 | 0.017468 | 0.018320 | -0.040916 | 0.502249 | 1001472 |
| original_env_reward | -1.966596 | 3.286034 | 1.000000 | -1.966596 | 3.286034 | -100.000000 | 148.092192 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_reward | 1.228927 | 1.228927 | 0.190857 | 1.419531 | 13896 |
| soft_landing_proxy | 0.145330 | 0.145330 | 0.000000 | 2.000000 | 13896 |
| stability_penalty | -0.115330 | 0.115330 | -0.303906 | -0.062723 | 13896 |
| total_reward | 1.258927 | 1.258927 | 0.005005 | 3.107507 | 13896 |
