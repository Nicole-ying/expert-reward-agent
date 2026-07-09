# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_reward | 0.033350 | 0.033350 | 0.903560 | 0.036909 | 0.036909 | 0.000000 | 0.085046 | 1001472 |
| component.soft_landing_bonus | 0.009332 | 0.009332 | 0.007073 | 1.319444 | 1.319444 | 0.000000 | 1.914399 | 1001472 |
| component.stability_penalty | -0.013830 | 0.013830 | 1.000000 | -0.013830 | 0.013830 | -0.072292 | -0.000000 | 1001472 |
| component.total_reward | 0.028851 | 0.031927 | 1.000000 | 0.028851 | 0.031927 | -0.072292 | 1.915781 | 1001472 |
| generated_reward | 0.028851 | 0.031927 | 1.000000 | 0.028851 | 0.031927 | -0.072292 | 1.915781 | 1001472 |
| original_env_reward | -1.486794 | 2.379492 | 1.000000 | -1.486794 | 2.379492 | -100.000000 | 132.122254 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_reward | 2.328731 | 2.328731 | 0.323020 | 3.033251 | 14342 |
| soft_landing_bonus | 0.651626 | 0.651626 | 0.000000 | 4.471959 | 14342 |
| stability_penalty | -0.965726 | 0.965726 | -2.596623 | -0.630696 | 14342 |
| total_reward | 2.014632 | 2.017165 | -1.697784 | 5.399999 | 14342 |
