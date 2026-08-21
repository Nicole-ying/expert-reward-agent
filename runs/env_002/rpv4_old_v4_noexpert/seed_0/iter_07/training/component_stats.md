# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.039026 | 0.039026 | 1.000000 | -0.039026 | 0.039026 | -0.080000 | -0.000055 | 1015808 |
| component.forward_progress | 0.255825 | 0.262067 | 0.999999 | 0.255825 | 0.262067 | -0.627989 | 0.809377 | 1015808 |
| component.torso_stability | -0.014881 | 0.014881 | 1.000000 | -0.014881 | 0.014881 | -1.538272 | -0.000000 | 1015808 |
| component.total_reward | 0.201339 | 0.222439 | 1.000000 | 0.201339 | 0.222439 | -1.756446 | 0.773977 | 1015808 |
| component.vertical_velocity_penalty | -0.000579 | 0.000579 | 0.999935 | -0.000579 | 0.000579 | -0.017750 | -0.000000 | 1015808 |
| generated_reward | 0.201339 | 0.222439 | 1.000000 | 0.201339 | 0.222439 | -1.756446 | 0.773977 | 1015808 |
| original_env_reward | -0.262997 | 0.480597 | 1.000000 | -0.262997 | 0.480597 | -100.000000 | 0.868342 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -10.736735 | 10.736735 | -67.186548 | -1.364096 | 3670 |
| forward_progress | 70.162541 | 70.749665 | -41.914421 | 612.893862 | 3670 |
| torso_stability | -4.092477 | 4.092477 | -48.733254 | -0.136729 | 3670 |
| total_reward | 55.173692 | 58.580235 | -138.103800 | 533.137342 | 3670 |
| vertical_velocity_penalty | -0.159637 | 0.159637 | -1.005833 | -0.046925 | 3670 |
