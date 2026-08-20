# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.descending_penalty | -0.024079 | 0.024079 | 0.111522 | -0.215908 | 0.215908 | -0.808558 | 0.000000 | 1003520 |
| component.landing_approach | 0.040215 | 0.040215 | 1.000000 | 0.040215 | 0.040215 | 0.004554 | 0.050000 | 1003520 |
| component.lateral_drift_penalty | -0.003290 | 0.003290 | 0.999782 | -0.003291 | 0.003291 | -0.323276 | -0.000000 | 1003520 |
| component.progress | 0.024313 | 0.027308 | 0.999904 | 0.024315 | 0.027311 | -0.295573 | 0.379662 | 1003520 |
| component.stability_penalty | -0.008840 | 0.008840 | 0.999938 | -0.008840 | 0.008840 | -7.456011 | -0.000000 | 1003520 |
| component.total_reward | 0.028320 | 0.067880 | 1.000000 | 0.028320 | 0.067880 | -7.986665 | 0.097558 | 1003520 |
| generated_reward | 0.028320 | 0.067880 | 1.000000 | 0.028320 | 0.067880 | -7.986665 | 0.097558 | 1003520 |
| original_env_reward | -0.056503 | 1.747743 | 1.000000 | -0.056503 | 1.747743 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| descending_penalty | -11.609074 | 11.609074 | -28.999873 | 0.000000 | 2081 |
| landing_approach | 19.329132 | 19.329132 | 0.725604 | 45.530969 | 2081 |
| lateral_drift_penalty | -1.585251 | 1.585251 | -12.687833 | -0.002858 | 2081 |
| progress | 11.697787 | 11.702636 | -2.407243 | 14.213753 | 2081 |
| stability_penalty | -4.261146 | 4.261146 | -87.693985 | -0.018896 | 2081 |
| total_reward | 13.571448 | 31.049558 | -95.149451 | 58.413890 | 2081 |
