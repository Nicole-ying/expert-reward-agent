# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.gated_forward | 0.172183 | 0.172183 | 0.542010 | 0.317675 | 0.317675 | 0.000000 | 0.852924 | 1001472 |
| component.lateral_penalty | 0.302190 | 0.302190 | 1.000000 | 0.302190 | 0.302190 | 0.000001 | 2.753214 | 1001472 |
| component.total_reward | -1.448965 | 1.559476 | 1.000000 | -1.448965 | 1.559476 | -7.058507 | 0.825503 | 1001472 |
| component.upright_penalty | 1.318958 | 1.318958 | 0.999984 | 1.318979 | 1.318979 | 0.000000 | 5.000000 | 1001472 |
| generated_reward | -1.448965 | 1.559476 | 1.000000 | -1.448965 | 1.559476 | -7.058507 | 0.825503 | 1001472 |
| original_env_reward | -1.257167 | 1.412872 | 1.000000 | -1.257167 | 1.412872 | -7.273310 | 4.852993 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| gated_forward | 3.478907 | 3.478907 | 0.000000 | 148.723748 | 49566 |
| lateral_penalty | 6.105683 | 6.105683 | 0.214898 | 245.908043 | 49566 |
| total_reward | -29.276081 | 29.915235 | -5073.959319 | 15.890589 | 49566 |
| upright_penalty | 26.649305 | 26.649305 | 0.000054 | 4964.474239 | 49566 |
