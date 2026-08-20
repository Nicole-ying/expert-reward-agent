# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.balance_angle_penalty | -0.002312 | 0.002312 | 0.066294 | -0.034879 | 0.034879 | -2.968704 | -0.000000 | 1015808 |
| component.balance_angular_vel_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000000 | -0.000000 | 1015808 |
| component.energy_penalty | -0.054465 | 0.054465 | 1.000000 | -0.054465 | 0.054465 | -0.200000 | -0.000018 | 1015808 |
| component.forward_progress_reward | 0.179102 | 0.179102 | 0.999984 | 0.179104 | 0.179104 | 0.000000 | 1.127103 | 1015808 |
| component.gate_factor | 0.788734 | 0.788734 | 0.981932 | 0.803248 | 0.803248 | 0.000000 | 1.000000 | 1015808 |
| component.gated_forward_reward | 0.150795 | 0.150795 | 0.981914 | 0.153573 | 0.153573 | 0.000000 | 1.055750 | 1015808 |
| component.total_reward | 0.094018 | 0.162839 | 1.000000 | 0.094018 | 0.162839 | -3.034223 | 1.043502 | 1015808 |
| generated_reward | 0.094018 | 0.162839 | 1.000000 | 0.094018 | 0.162839 | -3.034223 | 1.043502 | 1015808 |
| original_env_reward | -0.704165 | 1.010161 | 1.000000 | -0.704165 | 1.010161 | -100.000000 | 0.736214 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| balance_angle_penalty | -0.280759 | 0.280759 | -106.214062 | 0.000000 | 8366 |
| balance_angular_vel_penalty | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 8366 |
| energy_penalty | -6.607782 | 6.607782 | -168.011469 | -1.481174 | 8366 |
| forward_progress_reward | 21.704324 | 21.704324 | 0.008238 | 373.380961 | 8366 |
| gate_factor | 95.652829 | 95.652829 | 5.613057 | 1279.399779 | 8366 |
| gated_forward_reward | 18.269588 | 18.269588 | 0.001280 | 347.768111 | 8366 |
| total_reward | 11.381047 | 18.043931 | -186.239031 | 316.557237 | 8366 |
