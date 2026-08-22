# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.001107 | 0.001107 | 0.004064 | -0.272475 | 0.272475 | -1.927013 | -0.000000 | 1015808 |
| component.energy_penalty | -0.019497 | 0.019497 | 1.000000 | -0.019497 | 0.019497 | -0.040000 | -0.000038 | 1015808 |
| component.gated_forward_reward | 0.250347 | 0.250347 | 0.929303 | 0.269393 | 0.269393 | 0.000000 | 0.767130 | 1015808 |
| component.total_reward | 0.229743 | 0.236593 | 1.000000 | 0.229743 | 0.236593 | -1.959126 | 0.754923 | 1015808 |
| generated_reward | 0.229743 | 0.236593 | 1.000000 | 0.229743 | 0.236593 | -1.959126 | 0.754923 | 1015808 |
| original_env_reward | -0.389686 | 0.693572 | 1.000000 | -0.389686 | 0.693572 | -100.000000 | 0.763988 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.209807 | 0.209807 | -94.469662 | 0.000000 | 5361 |
| energy_penalty | -3.682480 | 3.682480 | -34.278737 | -0.658530 | 5361 |
| gated_forward_reward | 47.183149 | 47.183149 | 0.000140 | 471.019714 | 5361 |
| total_reward | 43.290862 | 43.792479 | -97.277696 | 449.811790 | 5361 |
