# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.034926 | 0.034926 | 1.000000 | -0.034926 | 0.034926 | -0.080000 | -0.000031 | 1015808 |
| component.total_reward | 0.273018 | 0.285364 | 1.000000 | 0.273018 | 0.285364 | -0.353844 | 0.737652 | 1015808 |
| component.velocity_gated | 0.308370 | 0.310570 | 1.000000 | 0.308370 | 0.310570 | -0.293595 | 0.766781 | 1015808 |
| component.vertical_penalty | -0.000426 | 0.000426 | 0.999920 | -0.000426 | 0.000426 | -0.011763 | -0.000000 | 1015808 |
| generated_reward | 0.273018 | 0.285364 | 1.000000 | 0.273018 | 0.285364 | -0.353844 | 0.737652 | 1015808 |
| original_env_reward | -0.494604 | 0.870321 | 1.000000 | -0.494604 | 0.870321 | -100.000000 | 0.757327 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -5.216391 | 5.216391 | -67.354176 | -1.343346 | 6786 |
| total_reward | 40.695061 | 41.416329 | -76.181128 | 469.023107 | 6786 |
| velocity_gated | 45.975174 | 46.005133 | -13.924310 | 498.353028 | 6786 |
| vertical_penalty | -0.063723 | 0.063723 | -0.359566 | -0.029472 | 6786 |
