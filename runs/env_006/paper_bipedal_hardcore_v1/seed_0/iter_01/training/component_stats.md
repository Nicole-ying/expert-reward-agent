# Reward Component Training Statistics

- steps_seen: 1440000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.gated_forward_speed | 0.237689 | 0.237689 | 0.989053 | 0.240320 | 0.240320 | 0.000000 | 1.884688 | 1440000 |
| component.posture_hinge_penalty | -0.008773 | 0.008773 | 1.000000 | -0.008773 | 0.008773 | -1.286095 | -0.000000 | 1440000 |
| component.total_reward | 0.228916 | 0.237625 | 1.000000 | 0.228916 | 0.237625 | -1.286095 | 1.883410 | 1440000 |
| generated_reward | 0.228916 | 0.237625 | 1.000000 | 0.228916 | 0.237625 | -1.286095 | 1.883410 | 1440000 |
| original_env_reward | -0.555815 | 0.848764 | 1.000000 | -0.555815 | 0.848764 | -100.000000 | 0.808955 | 1440000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| gated_forward_speed | 35.062737 | 35.062737 | 0.009461 | 178.424252 | 9756 |
| posture_hinge_penalty | -1.294575 | 1.294575 | -105.963312 | -0.214416 | 9756 |
| total_reward | 33.768162 | 34.154084 | -103.905241 | 177.328443 | 9756 |
