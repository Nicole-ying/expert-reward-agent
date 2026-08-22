# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.083275 | 0.083275 | 1.000000 | -0.083275 | 0.083275 | -0.200000 | -0.000030 | 1015808 |
| component.forward_progress_reward | 0.213521 | 0.219007 | 1.000000 | 0.213521 | 0.219007 | -0.380716 | 0.600472 | 1015808 |
| component.total_reward | 0.123480 | 0.178716 | 1.000000 | 0.123480 | 0.178716 | -8.002140 | 0.580918 | 1015808 |
| component.upright_stability_penalty | -0.006767 | 0.006767 | 0.436839 | -0.015490 | 0.015490 | -7.779058 | -0.000000 | 1015808 |
| generated_reward | 0.123480 | 0.178716 | 1.000000 | 0.123480 | 0.178716 | -8.002140 | 0.580918 | 1015808 |
| original_env_reward | -0.389101 | 0.633640 | 1.000000 | -0.389101 | 0.633640 | -100.000000 | 0.908991 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -16.734452 | 16.734452 | -168.024613 | -2.853675 | 5033 |
| forward_progress_reward | 42.764552 | 43.186300 | -16.812655 | 477.968035 | 5033 |
| total_reward | 24.666589 | 32.115938 | -231.943986 | 370.411256 | 5033 |
| upright_stability_penalty | -1.363510 | 1.363510 | -107.347674 | -0.000000 | 5033 |
