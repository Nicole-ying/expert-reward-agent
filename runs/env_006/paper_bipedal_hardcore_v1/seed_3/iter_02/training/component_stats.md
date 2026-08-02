# Reward Component Training Statistics

- steps_seen: 1040000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angular_penalty | -0.000135 | 0.000135 | 0.998612 | -0.000136 | 0.000136 | -0.005519 | -0.000000 | 1040000 |
| component.posture_gate | 0.649183 | 0.649183 | 1.000000 | 0.649183 | 0.649183 | 0.060668 | 0.999999 | 1040000 |
| component.progress_reward | 0.046214 | 0.046970 | 1.000000 | 0.046214 | 0.046970 | -0.063352 | 0.219676 | 1040000 |
| component.total_reward | 0.045611 | 0.046439 | 1.000000 | 0.045611 | 0.046439 | -0.064660 | 0.217542 | 1040000 |
| component.vertical_penalty | -0.000468 | 0.000468 | 0.999917 | -0.000468 | 0.000468 | -0.014831 | -0.000000 | 1040000 |
| generated_reward | 0.045611 | 0.046439 | 1.000000 | 0.045611 | 0.046439 | -0.064660 | 0.217542 | 1040000 |
| original_env_reward | -0.562564 | 0.787466 | 1.000000 | -0.562564 | 0.787466 | -100.000000 | 0.739771 | 1040000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angular_penalty | -0.021095 | 0.021095 | -0.177044 | -0.003254 | 6669 |
| posture_gate | 101.105014 | 101.105014 | 10.982520 | 1375.501797 | 6669 |
| progress_reward | 7.194614 | 7.200114 | -1.553137 | 48.097773 | 6669 |
| total_reward | 7.100546 | 7.108789 | -1.657756 | 47.711333 | 6669 |
| vertical_penalty | -0.072973 | 0.072973 | -0.329246 | -0.020167 | 6669 |
