# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_delta_gated | 0.001079 | 0.001332 | 0.999994 | 0.001079 | 0.001332 | -0.039432 | 0.006105 | 1003520 |
| component.contact_alignment | 0.001747 | 0.001747 | 0.017473 | 0.100000 | 0.100000 | 0.000000 | 0.100000 | 1003520 |
| component.energy_penalty | -0.000389 | 0.000389 | 0.007781 | -0.050000 | 0.050000 | -0.050000 | -0.000000 | 1003520 |
| component.soft_landing_bonus | 0.004538 | 0.004538 | 0.017473 | 0.259680 | 0.259680 | 0.000000 | 0.919823 | 1003520 |
| component.stability_penalty | -0.004705 | 0.004705 | 1.000000 | -0.004705 | 0.004705 | -0.434170 | -0.000000 | 1003520 |
| component.total_reward | 0.002270 | 0.009712 | 1.000000 | 0.002270 | 0.009712 | -0.465281 | 1.019417 | 1003520 |
| generated_reward | 0.002270 | 0.009712 | 1.000000 | 0.002270 | 0.009712 | -0.465281 | 1.019417 | 1003520 |
| original_env_reward | -1.594482 | 2.376695 | 1.000000 | -1.594482 | 2.376695 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_delta_gated | 0.075468 | 0.075483 | -0.026568 | 0.294007 | 14348 |
| contact_alignment | 0.122212 | 0.122212 | 0.000000 | 46.800000 | 14348 |
| energy_penalty | -0.027209 | 0.027209 | -12.250000 | 0.000000 | 14348 |
| soft_landing_bonus | 0.317360 | 0.317360 | 0.000000 | 134.749095 | 14348 |
| stability_penalty | -0.329077 | 0.329077 | -8.129647 | -0.039786 | 14348 |
| total_reward | 0.158754 | 0.586734 | -9.005833 | 161.190438 | 14348 |
