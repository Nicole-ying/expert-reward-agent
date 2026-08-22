# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.000000 | 0.000000 | 0.996607 | -0.000000 | 0.000000 | -0.000012 | -0.000000 | 1015808 |
| component.gait_cycle_reward | 0.133353 | 0.133353 | 0.391508 | 0.340614 | 0.340614 | 0.000000 | 0.600000 | 1015808 |
| component.tilt_penalty | -0.001019 | 0.001019 | 0.019193 | -0.053095 | 0.053095 | -0.477406 | -0.000000 | 1015808 |
| component.torque_penalty | -0.001127 | 0.001127 | 1.000000 | -0.001127 | 0.001127 | -0.002000 | -0.000001 | 1015808 |
| component.total_reward | 0.131207 | 0.134306 | 1.000000 | 0.131207 | 0.134306 | -0.478555 | 0.599992 | 1015808 |
| generated_reward | 0.131207 | 0.134306 | 1.000000 | 0.131207 | 0.134306 | -0.478555 | 0.599992 | 1015808 |
| original_env_reward | -0.079180 | 0.096792 | 1.000000 | -0.079180 | 0.096792 | -100.000000 | 0.512788 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.000178 | 0.000178 | -0.000568 | -0.000016 | 818 |
| gait_cycle_reward | 164.318215 | 164.318215 | 0.000000 | 269.700000 | 818 |
| tilt_penalty | -1.265462 | 1.265462 | -54.417735 | 0.000000 | 818 |
| torque_penalty | -1.389720 | 1.389720 | -1.989961 | -0.033849 | 818 |
| total_reward | 161.662855 | 162.056792 | -14.235542 | 267.746023 | 818 |
