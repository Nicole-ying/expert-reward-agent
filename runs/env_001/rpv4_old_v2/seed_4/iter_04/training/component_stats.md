# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_progress | 0.012453 | 0.015717 | 0.999739 | 0.012456 | 0.015721 | -0.178047 | 0.183747 | 1003520 |
| component.contact_completion | 1.765371 | 1.765371 | 0.489001 | 3.610160 | 3.610160 | 0.000000 | 4.999997 | 1003520 |
| component.orientation_penalty | -0.028582 | 0.028582 | 0.999986 | -0.028582 | 0.028582 | -14.819149 | -0.000000 | 1003520 |
| component.soft_landing_speed_penalty | -0.051999 | 0.051999 | 0.998505 | -0.052077 | 0.052077 | -1.958091 | -0.000000 | 1003520 |
| component.total_reward | 1.697243 | 1.804833 | 1.000000 | 1.697243 | 1.804833 | -14.941139 | 4.999960 | 1003520 |
| generated_reward | 1.697243 | 1.804833 | 1.000000 | 1.697243 | 1.804833 | -14.941139 | 4.999960 | 1003520 |
| original_env_reward | -0.144522 | 1.761310 | 1.000000 | -0.144522 | 1.761310 | -100.000000 | 122.565771 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_progress | 4.457592 | 4.723145 | -28.868921 | 7.102259 | 2798 |
| contact_completion | 630.658404 | 630.658404 | 0.000000 | 4196.730089 | 2798 |
| orientation_penalty | -10.232256 | 10.232256 | -266.869648 | -0.060739 | 2798 |
| soft_landing_speed_penalty | -18.611414 | 18.611414 | -113.488475 | -0.001085 | 2798 |
| total_reward | 606.272325 | 625.830147 | -209.502988 | 4163.423432 | 2798 |
