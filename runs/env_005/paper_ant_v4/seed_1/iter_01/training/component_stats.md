# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 0.054176 | 0.564544 | 1.000000 | 0.054176 | 0.564544 | -1.626190 | 1.646044 | 1015808 |
| component.height_penalty | -0.311187 | 0.311187 | 0.428849 | -0.725633 | 0.725633 | -1.999965 | 0.000000 | 1015808 |
| component.lateral_drift_penalty | -0.043934 | 0.043934 | 0.999995 | -0.043934 | 0.043934 | -2.036954 | -0.000000 | 1015808 |
| component.total_reward | -0.263640 | 1.298247 | 1.000000 | -0.263640 | 1.298247 | -4.387332 | 2.552073 | 1015808 |
| component.upright_stability | 0.037305 | 0.940319 | 1.000000 | 0.037305 | 0.940319 | -1.635781 | 0.999982 | 1015808 |
| generated_reward | -0.263640 | 1.298247 | 1.000000 | -0.263640 | 1.298247 | -4.387332 | 2.552073 | 1015808 |
| original_env_reward | -0.883411 | 1.026641 | 1.000000 | -0.883411 | 1.026641 | -6.022133 | 4.000042 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 8.932481 | 20.477998 | -132.652812 | 230.527625 | 6146 |
| height_penalty | -51.273696 | 51.273696 | -679.876881 | 0.000000 | 6146 |
| lateral_drift_penalty | -7.249382 | 7.249382 | -64.623233 | -0.037027 | 6146 |
| total_reward | -43.340932 | 188.381013 | -1632.716284 | 1143.011366 | 6146 |
| upright_stability | 6.249665 | 139.283623 | -940.017453 | 963.858505 | 6146 |
