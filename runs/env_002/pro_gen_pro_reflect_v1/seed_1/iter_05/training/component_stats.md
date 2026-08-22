# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_efficiency | -0.038273 | 0.038273 | 1.000000 | -0.038273 | 0.038273 | -0.080000 | -0.000036 | 1015808 |
| component.stability_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1015808 |
| component.total_reward | 0.231359 | 0.240911 | 1.000000 | 0.231359 | 0.240911 | -0.317164 | 0.669147 | 1015808 |
| component.velocity_gated | 0.269631 | 0.271945 | 0.992859 | 0.271571 | 0.273901 | -0.263116 | 0.696445 | 1015808 |
| generated_reward | 0.231359 | 0.240911 | 1.000000 | 0.231359 | 0.240911 | -0.317164 | 0.669147 | 1015808 |
| original_env_reward | -0.048101 | 0.365078 | 1.000000 | -0.048101 | 0.365078 | -100.000000 | 0.922388 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_efficiency | -17.676612 | 17.676612 | -67.354176 | -1.435114 | 2191 |
| stability_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2191 |
| total_reward | 106.624435 | 108.744136 | -81.931548 | 460.355016 | 2191 |
| velocity_gated | 124.301047 | 124.411774 | -16.503620 | 504.784849 | 2191 |
