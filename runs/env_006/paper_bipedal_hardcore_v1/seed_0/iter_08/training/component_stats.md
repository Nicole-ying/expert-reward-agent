# Reward Component Training Statistics

- steps_seen: 1360000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.009531 | 0.009531 | 1.000000 | -0.009531 | 0.009531 | -0.020000 | -0.000003 | 1360000 |
| component.forward_reward | 0.246744 | 0.246744 | 0.914510 | 0.269810 | 0.269810 | 0.000000 | 0.827665 | 1360000 |
| component.gated_forward | 0.232526 | 0.232526 | 0.904373 | 0.257112 | 0.257112 | 0.000000 | 0.827665 | 1360000 |
| component.stability_gate | 0.926766 | 0.926766 | 0.984874 | 0.941000 | 0.941000 | 0.000000 | 1.000000 | 1360000 |
| component.total_reward | 0.222994 | 0.225345 | 1.000000 | 0.222994 | 0.225345 | -0.020000 | 0.818632 | 1360000 |
| generated_reward | 0.222994 | 0.225345 | 1.000000 | 0.222994 | 0.225345 | -0.020000 | 0.818632 | 1360000 |
| original_env_reward | -0.483141 | 0.740890 | 1.000000 | -0.483141 | 0.740890 | -100.000000 | 0.795626 | 1360000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -1.639090 | 1.639090 | -16.797725 | -0.351922 | 7903 |
| forward_reward | 42.424157 | 42.424157 | 0.000869 | 301.242458 | 7903 |
| gated_forward | 39.978893 | 39.978893 | 0.000869 | 289.989369 | 7903 |
| stability_gate | 159.366452 | 159.366452 | 9.357596 | 1599.347528 | 7903 |
| total_reward | 38.339803 | 38.358556 | -1.543404 | 283.357129 | 7903 |
