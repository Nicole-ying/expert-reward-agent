# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_distance_progress | 0.013105 | 0.015623 | 0.999489 | 0.013111 | 0.015631 | -0.138806 | 0.183074 | 1003520 |
| component.landing_contact_success | 0.915204 | 0.915204 | 0.457602 | 2.000000 | 2.000000 | 0.000000 | 2.000000 | 1003520 |
| component.orientation_penalty | -0.013961 | 0.013961 | 0.999881 | -0.013963 | 0.013963 | -8.943005 | -0.000000 | 1003520 |
| component.soft_landing | -0.020541 | 0.020541 | 0.999684 | -0.020548 | 0.020548 | -2.832249 | -0.000000 | 1003520 |
| component.total_reward | 0.893806 | 0.938128 | 1.000000 | 0.893806 | 0.938128 | -9.040453 | 2.007277 | 1003520 |
| generated_reward | 0.893806 | 0.938128 | 1.000000 | 0.893806 | 0.938128 | -9.040453 | 2.007277 | 1003520 |
| original_env_reward | -0.070383 | 1.509206 | 1.000000 | -0.070383 | 1.509206 | -100.000000 | 127.277999 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_distance_progress | 5.044974 | 5.050043 | -1.750956 | 7.082929 | 2603 |
| landing_contact_success | 351.795620 | 351.795620 | 0.000000 | 1768.000000 | 2603 |
| orientation_penalty | -5.376823 | 5.376823 | -144.156827 | -0.004190 | 2603 |
| soft_landing | -7.910777 | 7.910777 | -59.372611 | -0.965098 | 2603 |
| total_reward | 343.552994 | 354.352719 | -149.627391 | 1763.608796 | 2603 |
