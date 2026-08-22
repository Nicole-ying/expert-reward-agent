# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | 0.001467 | 0.001467 | 0.999378 | 0.001468 | 0.001468 | 0.000000 | 1.125821 | 1003520 |
| component.progress_reward | 0.005349 | 0.005754 | 0.999449 | 0.005351 | 0.005757 | -0.051618 | 0.074477 | 1003520 |
| component.soft_landing_proxy | 0.252043 | 0.252043 | 0.593522 | 0.424657 | 0.424657 | 0.000000 | 0.499120 | 1003520 |
| component.total_reward | 0.249861 | 0.256474 | 1.000000 | 0.249861 | 0.256474 | -1.122223 | 0.499120 | 1003520 |
| component.velocity_penalty | 0.006063 | 0.006063 | 0.999351 | 0.006067 | 0.006067 | 0.000000 | 0.295644 | 1003520 |
| generated_reward | 0.249861 | 0.256474 | 1.000000 | 0.249861 | 0.256474 | -1.122223 | 0.499120 | 1003520 |
| original_env_reward | 0.042789 | 1.213409 | 1.000000 | 0.042789 | 1.213409 | -100.000000 | 130.659578 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | 0.669385 | 0.669385 | 0.001842 | 15.648166 | 2196 |
| progress_reward | 2.439948 | 2.439948 | 0.095738 | 2.838708 | 2196 |
| soft_landing_proxy | 114.997517 | 114.997517 | 0.000000 | 403.052371 | 2196 |
| total_reward | 114.000297 | 116.427845 | -18.988316 | 404.296954 | 2196 |
| velocity_penalty | 2.767782 | 2.767782 | 0.812135 | 9.368160 | 2196 |
