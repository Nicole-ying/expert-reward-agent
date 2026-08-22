# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.balance_angle_penalty | -0.001655 | 0.001655 | 0.031661 | -0.052270 | 0.052270 | -2.968704 | -0.000000 | 1015808 |
| component.balance_angular_vel_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1015808 |
| component.forward_progress_reward | 0.153027 | 0.153027 | 0.999989 | 0.153028 | 0.153028 | 0.000000 | 0.818883 | 1015808 |
| component.gate_factor | 0.833423 | 0.833423 | 0.990408 | 0.841495 | 0.841495 | 0.000000 | 0.999999 | 1015808 |
| component.gated_forward_reward | 0.138040 | 0.138040 | 0.990396 | 0.139378 | 0.139378 | 0.000000 | 0.752029 | 1015808 |
| component.total_reward | 0.136385 | 0.139647 | 0.999989 | 0.136386 | 0.139648 | -2.968704 | 0.752029 | 1015808 |
| generated_reward | 0.136385 | 0.139647 | 0.999989 | 0.136386 | 0.139648 | -2.968704 | 0.752029 | 1015808 |
| original_env_reward | 0.086041 | 0.217408 | 1.000000 | 0.086041 | 0.217408 | -100.000000 | 0.748697 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| balance_angle_penalty | -1.607138 | 1.607138 | -61.961846 | 0.000000 | 1046 |
| balance_angular_vel_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1046 |
| forward_progress_reward | 147.429461 | 147.429461 | 0.011916 | 373.882524 | 1046 |
| gate_factor | 806.116918 | 806.116918 | 5.926322 | 1409.064674 | 1046 |
| gated_forward_reward | 132.950928 | 132.950928 | 0.001997 | 348.198139 | 1046 |
| total_reward | 131.343790 | 134.142611 | -61.722622 | 348.198139 | 1046 |
