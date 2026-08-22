# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.001061 | 0.001061 | 0.999858 | -0.001061 | 0.001061 | -0.349247 | -0.000000 | 1003520 |
| component.progress_delta | 0.011445 | 0.012268 | 0.997071 | 0.011478 | 0.012304 | -0.038366 | 0.042294 | 1003520 |
| component.soft_landing | 0.118352 | 0.118352 | 0.144400 | 0.819617 | 0.819617 | 0.000000 | 1.000000 | 1003520 |
| component.total_reward | 0.128737 | 0.130751 | 1.000000 | 0.128737 | 0.130751 | -0.365657 | 1.000000 | 1003520 |
| generated_reward | 0.128737 | 0.130751 | 1.000000 | 0.128737 | 0.130751 | -0.365657 | 1.000000 | 1003520 |
| original_env_reward | -0.841573 | 2.407750 | 1.000000 | -0.841573 | 2.407750 | -100.000000 | 129.433189 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -0.106757 | 0.106757 | -2.413226 | -0.000295 | 9963 |
| progress_delta | 1.152260 | 1.152409 | -0.369975 | 1.407289 | 9963 |
| soft_landing | 11.875213 | 11.875213 | 0.000000 | 841.344606 | 9963 |
| total_reward | 12.920716 | 12.925682 | -1.763065 | 842.655051 | 9963 |
