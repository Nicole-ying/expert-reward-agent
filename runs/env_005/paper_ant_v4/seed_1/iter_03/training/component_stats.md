# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 2.571244 | 16.725589 | 1.000000 | 2.571244 | 16.725589 | -47.879611 | 50.383971 | 1015808 |
| component.height_penalty | -0.377019 | 0.377019 | 0.519184 | -0.726176 | 0.726176 | -1.999936 | 0.000000 | 1015808 |
| component.lateral_drift_penalty | -0.045050 | 0.045050 | 0.999996 | -0.045050 | 0.045050 | -2.048498 | -0.000000 | 1015808 |
| component.total_reward | 1.114302 | 13.707164 | 1.000000 | 1.114302 | 13.707164 | -20.000000 | 20.000000 | 1015808 |
| component.upright_stability | -0.171126 | 0.936811 | 1.000000 | -0.171126 | 0.936811 | -1.818805 | 0.999981 | 1015808 |
| generated_reward | 1.114302 | 13.707164 | 1.000000 | 1.114302 | 13.707164 | -20.000000 | 20.000000 | 1015808 |
| original_env_reward | -0.854581 | 1.012522 | 1.000000 | -0.854581 | 1.012522 | -6.425297 | 4.771858 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 348.526839 | 585.405231 | -3256.364054 | 6115.812678 | 7476 |
| height_penalty | -50.955293 | 50.955293 | -674.157027 | 0.000000 | 7476 |
| lateral_drift_penalty | -6.108006 | 6.108006 | -45.879680 | -0.028820 | 7476 |
| total_reward | 151.397350 | 425.039856 | -2993.032312 | 3906.258918 | 7476 |
| upright_stability | -22.923810 | 116.722571 | -939.152620 | 549.635869 | 7476 |
