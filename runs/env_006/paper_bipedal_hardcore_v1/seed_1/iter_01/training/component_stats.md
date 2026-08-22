# Reward Component Training Statistics

- steps_seen: 1360000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_efficiency_penalty | -0.018466 | 0.018466 | 1.000000 | -0.018466 | 0.018466 | -0.040000 | -0.000028 | 1360000 |
| component.progress_reward | 0.218175 | 0.227327 | 0.999999 | 0.218175 | 0.227328 | -0.511104 | 0.805637 | 1360000 |
| component.stability_penalty | -0.010029 | 0.010029 | 0.185138 | -0.054168 | 0.054168 | -1.414533 | -0.000000 | 1360000 |
| component.total_reward | 0.189681 | 0.210429 | 1.000000 | 0.189681 | 0.210429 | -1.521515 | 0.783620 | 1360000 |
| generated_reward | 0.189681 | 0.210429 | 1.000000 | 0.189681 | 0.210429 | -1.521515 | 0.783620 | 1360000 |
| original_env_reward | -0.472432 | 0.699719 | 1.000000 | -0.472432 | 0.699719 | -100.000000 | 0.788673 | 1360000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_efficiency_penalty | -3.311287 | 3.311287 | -40.071902 | -0.584062 | 7575 |
| progress_reward | 39.101397 | 39.298863 | -16.810170 | 354.838809 | 7575 |
| stability_penalty | -1.800100 | 1.800100 | -79.275526 | 0.000000 | 7575 |
| total_reward | 33.990010 | 35.229485 | -117.777979 | 334.394209 | 7575 |
