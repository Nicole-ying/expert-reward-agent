# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_reward | 0.033498 | 0.033498 | 0.904703 | 0.037027 | 0.037027 | 0.000000 | 0.084733 | 1001472 |
| component.soft_landing_bonus | 0.004870 | 0.004870 | 0.004870 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1001472 |
| component.stability_penalty | -0.013818 | 0.013818 | 1.000000 | -0.013818 | 0.013818 | -0.077605 | -0.000000 | 1001472 |
| component.total_reward | 0.024550 | 0.027610 | 1.000000 | 0.024550 | 0.027610 | -0.077605 | 1.000321 | 1001472 |
| generated_reward | 0.024550 | 0.027610 | 1.000000 | 0.024550 | 0.027610 | -0.077605 | 1.000321 | 1001472 |
| original_env_reward | -1.522962 | 2.466030 | 1.000000 | -1.522962 | 2.466030 | -100.000000 | 132.122254 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_reward | 2.331103 | 2.331103 | 0.528538 | 3.036958 | 14391 |
| soft_landing_bonus | 0.338892 | 0.338892 | 0.000000 | 1.000000 | 14391 |
| stability_penalty | -0.961567 | 0.961567 | -4.591704 | -0.635742 | 14391 |
| total_reward | 1.708429 | 1.710991 | -2.254551 | 3.276328 | 14391 |
