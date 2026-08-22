# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_velocity_reward | 0.341576 | 0.422308 | 0.660490 | 0.517156 | 0.639386 | -3.613758 | 4.796648 | 1003520 |
| component.orientation_reward | 0.163993 | 0.474579 | 1.000000 | 0.163993 | 0.474579 | -0.500000 | 0.500000 | 1003520 |
| component.stability_penalty | -0.219241 | 0.219241 | 1.000000 | -0.219241 | 0.219241 | -4.808648 | -0.000044 | 1003520 |
| component.total_reward | 0.286329 | 0.755348 | 1.000000 | 0.286329 | 0.755348 | -4.816473 | 4.779953 | 1003520 |
| generated_reward | 0.286329 | 0.755348 | 1.000000 | 0.286329 | 0.755348 | -4.816473 | 4.779953 | 1003520 |
| original_env_reward | -0.771123 | 1.001997 | 1.000000 | -0.771123 | 1.001997 | -5.402429 | 4.635524 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_velocity_reward | 96.587060 | 100.541064 | -76.315128 | 1065.563826 | 3540 |
| orientation_reward | 46.396610 | 119.278502 | -459.871299 | 484.535770 | 3540 |
| stability_penalty | -62.077996 | 62.077996 | -291.225106 | -1.144338 | 3540 |
| total_reward | 80.905674 | 179.353054 | -656.021742 | 1270.625380 | 3540 |
