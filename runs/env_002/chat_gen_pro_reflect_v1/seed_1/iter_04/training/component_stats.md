# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.000839 | 0.000839 | 0.002916 | -0.287602 | 0.287602 | -1.927013 | -0.000000 | 1015808 |
| component.energy_penalty | -0.019847 | 0.019847 | 1.000000 | -0.019847 | 0.019847 | -0.040000 | -0.000009 | 1015808 |
| component.gated_forward_reward | 0.248260 | 0.248260 | 0.923102 | 0.268941 | 0.268941 | 0.000000 | 0.659727 | 1015808 |
| component.total_reward | 0.227574 | 0.233842 | 1.000000 | 0.227574 | 0.233842 | -1.959126 | 0.650523 | 1015808 |
| generated_reward | 0.227574 | 0.233842 | 1.000000 | 0.227574 | 0.233842 | -1.959126 | 0.650523 | 1015808 |
| original_env_reward | -0.119019 | 0.479248 | 1.000000 | -0.119019 | 0.479248 | -100.000000 | 0.766785 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.278300 | 0.278300 | -94.469662 | 0.000000 | 3061 |
| energy_penalty | -6.563426 | 6.563426 | -33.818603 | -0.707540 | 3061 |
| gated_forward_reward | 81.909173 | 81.909173 | 0.000000 | 408.259279 | 3061 |
| total_reward | 75.067446 | 75.916575 | -97.283120 | 390.130770 | 3061 |
