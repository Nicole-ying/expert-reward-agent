# Reward Component Training Statistics

- steps_seen: 1200000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angular_penalty | -0.000219 | 0.000219 | 0.998788 | -0.000219 | 0.000219 | -0.004795 | -0.000000 | 1200000 |
| component.posture_penalty | -0.000916 | 0.000916 | 0.041487 | -0.022076 | 0.022076 | -0.226883 | -0.000000 | 1200000 |
| component.progress_reward | 0.069171 | 0.071652 | 1.000000 | 0.069171 | 0.071652 | -0.163382 | 0.256844 | 1200000 |
| component.total_reward | 0.066595 | 0.069986 | 1.000000 | 0.066595 | 0.069986 | -0.311373 | 0.248617 | 1200000 |
| component.vertical_penalty | -0.001441 | 0.001441 | 0.999966 | -0.001441 | 0.001441 | -0.041890 | -0.000000 | 1200000 |
| generated_reward | 0.066595 | 0.069986 | 1.000000 | 0.066595 | 0.069986 | -0.311373 | 0.248617 | 1200000 |
| original_env_reward | -0.496260 | 0.737902 | 1.000000 | -0.496260 | 0.737902 | -100.000000 | 0.934995 | 1200000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angular_penalty | -0.038026 | 0.038026 | -0.320668 | -0.003892 | 6894 |
| posture_penalty | -0.159379 | 0.159379 | -9.267837 | 0.000000 | 6894 |
| progress_reward | 12.028222 | 12.064647 | -5.846857 | 104.631555 | 6894 |
| total_reward | 11.580140 | 11.704714 | -9.997697 | 101.825034 | 6894 |
| vertical_penalty | -0.250677 | 0.250677 | -1.760209 | -0.044888 | 6894 |
