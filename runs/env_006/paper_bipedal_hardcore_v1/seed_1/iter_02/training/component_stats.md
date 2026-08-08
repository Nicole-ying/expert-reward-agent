# Reward Component Training Statistics

- steps_seen: 1120000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_efficiency_penalty | -0.018242 | 0.018242 | 1.000000 | -0.018242 | 0.018242 | -0.040000 | -0.000024 | 1120000 |
| component.angular_velocity_penalty | -0.000177 | 0.000177 | 0.998988 | -0.000177 | 0.000177 | -0.007108 | -0.000000 | 1120000 |
| component.progress_reward | 0.201583 | 0.210600 | 1.000000 | 0.201583 | 0.210600 | -0.537086 | 0.785685 | 1120000 |
| component.stability_penalty | -0.008274 | 0.008274 | 0.140950 | -0.058701 | 0.058701 | -1.414533 | -0.000000 | 1120000 |
| component.total_reward | 0.174890 | 0.195501 | 1.000000 | 0.174890 | 0.195501 | -1.522450 | 0.748920 | 1120000 |
| generated_reward | 0.174890 | 0.195501 | 1.000000 | 0.174890 | 0.195501 | -1.522450 | 0.748920 | 1120000 |
| original_env_reward | -0.522189 | 0.730029 | 1.000000 | -0.522189 | 0.730029 | -100.000000 | 0.739680 | 1120000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_efficiency_penalty | -3.062934 | 3.062934 | -33.814183 | -0.602012 | 6665 |
| angular_velocity_penalty | -0.029661 | 0.029661 | -0.272546 | -0.004943 | 6665 |
| progress_reward | 33.825078 | 34.090649 | -16.971963 | 277.617850 | 6665 |
| stability_penalty | -1.390311 | 1.390311 | -79.275526 | 0.000000 | 6665 |
| total_reward | 29.342172 | 30.681953 | -117.983755 | 258.620686 | 6665 |
