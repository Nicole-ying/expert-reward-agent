# Reward Component Training Statistics

- steps_seen: 4096
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.dist_gate | 0.187903 | 0.187903 | 1.000000 | 0.187903 | 0.187903 | 0.111515 | 0.924607 | 4096 |
| component.landing_proxy | 0.000793 | 0.000793 | 0.005859 | 0.135337 | 0.135337 | 0.000000 | 0.806006 | 4096 |
| component.progress_reward | 0.010409 | 0.011710 | 1.000000 | 0.010409 | 0.011710 | -0.024156 | 0.035022 | 4096 |
| component.stability_penalty | -0.002898 | 0.002898 | 1.000000 | -0.002898 | 0.002898 | -0.045537 | -0.000026 | 4096 |
| component.total_reward | 0.008304 | 0.010695 | 1.000000 | 0.008304 | 0.010695 | -0.041416 | 0.798835 | 4096 |
| generated_reward | 0.008304 | 0.010695 | 1.000000 | 0.008304 | 0.010695 | -0.041416 | 0.798835 | 4096 |
| original_env_reward | -1.964652 | 3.184120 | 1.000000 | -1.964652 | 3.184120 | -100.000000 | 99.817124 | 4096 |

## Completed Training-Episode Component Sums

| component | episode_count | mean_sum | min_sum | max_sum |
|---|---:|---:|---:|---:|
| dist_gate | 42 | 17.540424 | 11.912964 | 23.165392 |
| landing_proxy | 42 | 0.077335 | 0.000000 | 0.806006 |
| progress_reward | 42 | 0.963570 | 0.309187 | 1.397724 |
| stability_penalty | 42 | -0.272128 | -0.485892 | -0.189355 |
| total_reward | 42 | 0.768777 | 0.002452 | 1.874338 |
