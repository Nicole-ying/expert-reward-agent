# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.020139 | 0.020139 | 0.671307 | -0.030000 | 0.030000 | -0.030000 | -0.000000 | 1003520 |
| component.landing_proxy_reward | 1.192106 | 1.192106 | 0.377235 | 3.160113 | 3.160113 | 0.000000 | 3.995023 | 1003520 |
| component.orientation_stability_reward | -0.077774 | 0.077774 | 0.999975 | -0.077776 | 0.077776 | -32.553287 | -0.000000 | 1003520 |
| component.progress_reward | 0.020996 | 0.023103 | 0.999115 | 0.021015 | 0.023124 | -0.076020 | 0.101747 | 1003520 |
| component.total_reward | 1.115414 | 1.241675 | 1.000000 | 1.115414 | 1.241675 | -20.000000 | 3.995023 | 1003520 |
| generated_reward | 1.115414 | 1.241675 | 1.000000 | 1.115414 | 1.241675 | -20.000000 | 3.995023 | 1003520 |
| original_env_reward | -0.279048 | 1.963940 | 1.000000 | -0.279048 | 1.963940 | -100.000000 | 137.975729 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -3.514160 | 3.514160 | -27.300000 | -0.690000 | 5741 |
| landing_proxy_reward | 207.902859 | 207.902859 | 0.000000 | 3338.677006 | 5741 |
| orientation_stability_reward | -13.588160 | 13.588160 | -396.613708 | -0.040431 | 5741 |
| progress_reward | 3.667457 | 3.667457 | 0.328064 | 4.044612 | 5741 |
| total_reward | 194.507399 | 209.921988 | -372.387088 | 3324.356201 | 5741 |
