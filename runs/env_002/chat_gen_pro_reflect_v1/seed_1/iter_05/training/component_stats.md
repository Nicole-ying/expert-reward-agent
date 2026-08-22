# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.018273 | 0.018273 | 1.000000 | -0.018273 | 0.018273 | -0.040000 | -0.000020 | 1015808 |
| component.gated_forward_reward | 0.234366 | 0.234366 | 0.922508 | 0.254053 | 0.254053 | 0.000000 | 0.617167 | 1015808 |
| component.total_reward | 0.216093 | 0.220689 | 1.000000 | 0.216093 | 0.220689 | -0.040000 | 0.610473 | 1015808 |
| generated_reward | 0.216093 | 0.220689 | 1.000000 | 0.216093 | 0.220689 | -0.040000 | 0.610473 | 1015808 |
| original_env_reward | -0.394619 | 0.739243 | 1.000000 | -0.394619 | 0.739243 | -100.000000 | 0.826630 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -3.260733 | 3.260733 | -33.677088 | -0.712857 | 5670 |
| gated_forward_reward | 41.727071 | 41.727071 | 0.000785 | 408.467043 | 5670 |
| total_reward | 38.466338 | 38.630069 | -19.822963 | 389.002507 | 5670 |
