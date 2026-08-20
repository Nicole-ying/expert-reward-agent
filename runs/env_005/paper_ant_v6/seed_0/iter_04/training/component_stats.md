# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.311462 | 0.311462 | 1.000000 | -0.311462 | 0.311462 | -0.800000 | -0.002406 | 1001472 |
| component.forward_reward | 0.332788 | 0.332788 | 0.578563 | 0.575197 | 0.575197 | 0.000000 | 4.996019 | 1001472 |
| component.lateral_penalty | -0.180958 | 0.180958 | 0.999994 | -0.180959 | 0.180959 | -10.909788 | -0.000000 | 1001472 |
| component.total_reward | -0.159632 | 0.579402 | 1.000000 | -0.159632 | 0.579402 | -11.308373 | 4.822057 | 1001472 |
| generated_reward | -0.159632 | 0.579402 | 1.000000 | -0.159632 | 0.579402 | -11.308373 | 4.822057 | 1001472 |
| original_env_reward | -0.195267 | 1.047961 | 1.000000 | -0.195267 | 1.047961 | -6.325668 | 5.514893 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -10.434622 | 10.434622 | -419.309009 | -1.948748 | 29890 |
| forward_reward | 11.149353 | 11.149353 | 0.000000 | 102.569087 | 29890 |
| lateral_penalty | -6.062489 | 6.062489 | -255.657483 | -0.021711 | 29890 |
| total_reward | -5.347759 | 13.897219 | -667.680164 | 74.072330 | 29890 |
