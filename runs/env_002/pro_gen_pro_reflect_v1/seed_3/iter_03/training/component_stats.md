# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.023702 | 0.023702 | 1.000000 | -0.023702 | 0.023702 | -0.050000 | -0.000001 | 1015808 |
| component.angular_velocity_penalty | -0.001086 | 0.001086 | 0.999942 | -0.001086 | 0.001086 | -0.048881 | -0.000000 | 1015808 |
| component.forward_progress | 0.224068 | 0.230647 | 1.000000 | 0.224068 | 0.230647 | -0.416354 | 0.795617 | 1015808 |
| component.tilt_penalty | -0.033867 | 0.033867 | 0.999991 | -0.033867 | 0.033867 | -16.108232 | -0.000000 | 1015808 |
| component.total_reward | 0.165414 | 0.235249 | 1.000000 | 0.165414 | 0.235249 | -16.163588 | 0.782086 | 1015808 |
| generated_reward | 0.165414 | 0.235249 | 1.000000 | 0.165414 | 0.235249 | -16.163588 | 0.782086 | 1015808 |
| original_env_reward | -0.257526 | 0.518680 | 1.000000 | -0.257526 | 0.518680 | -100.000000 | 0.671976 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -6.131707 | 6.131707 | -42.730336 | -0.713724 | 3914 |
| angular_velocity_penalty | -0.281407 | 0.281407 | -3.847116 | -0.049195 | 3914 |
| forward_progress | 57.698724 | 58.191589 | -22.690580 | 497.344930 | 3914 |
| tilt_penalty | -8.786166 | 8.786166 | -537.865198 | -1.067466 | 3914 |
| total_reward | 42.499445 | 56.371981 | -568.777399 | 474.561757 | 3914 |
