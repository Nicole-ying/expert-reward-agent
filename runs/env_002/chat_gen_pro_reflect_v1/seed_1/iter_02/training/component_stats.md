# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.000935 | 0.000935 | 0.003053 | -0.306258 | 0.306258 | -2.245620 | -0.000000 | 1015808 |
| component.energy_penalty | -0.017401 | 0.017401 | 1.000000 | -0.017401 | 0.017401 | -0.040000 | -0.000046 | 1015808 |
| component.gated_forward_reward | 0.330219 | 0.330219 | 0.999993 | 0.330221 | 0.330221 | 0.000000 | 2.153918 | 1015808 |
| component.total_reward | 0.311883 | 0.325531 | 1.000000 | 0.311883 | 0.325531 | -2.259156 | 2.144039 | 1015808 |
| generated_reward | 0.311883 | 0.325531 | 1.000000 | 0.311883 | 0.325531 | -2.259156 | 2.144039 | 1015808 |
| original_env_reward | -0.572667 | 0.870923 | 1.000000 | -0.572667 | 0.870923 | -100.000000 | 0.775409 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.134691 | 0.134691 | -94.469662 | 0.000000 | 7051 |
| energy_penalty | -2.505303 | 2.505303 | -33.904943 | -0.550179 | 7051 |
| gated_forward_reward | 47.522374 | 47.522374 | 0.025252 | 366.647052 | 7051 |
| total_reward | 44.882380 | 46.057287 | -97.853536 | 360.075558 | 7051 |
