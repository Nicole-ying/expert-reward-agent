# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_progress | 0.010018 | 0.013813 | 0.997929 | 0.010039 | 0.013842 | -0.173332 | 0.186276 | 1003520 |
| component.contact_completion | 1.878291 | 1.878291 | 0.444682 | 4.223899 | 4.223899 | 0.000000 | 4.999997 | 1003520 |
| component.orientation_penalty | -0.020198 | 0.020198 | 0.999959 | -0.020198 | 0.020198 | -9.261202 | -0.000000 | 1003520 |
| component.soft_landing_speed_penalty | -0.079866 | 0.079866 | 0.996618 | -0.080137 | 0.080137 | -3.424292 | -0.000000 | 1003520 |
| component.total_reward | 1.788245 | 1.948121 | 1.000000 | 1.788245 | 1.948121 | -10.842813 | 4.999965 | 1003520 |
| generated_reward | 1.788245 | 1.948121 | 1.000000 | 1.788245 | 1.948121 | -10.842813 | 4.999965 | 1003520 |
| original_env_reward | -0.145307 | 1.471298 | 1.000000 | -0.145307 | 1.471298 | -100.000000 | 136.926404 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_progress | 3.953086 | 4.716795 | -42.314067 | 7.100073 | 2537 |
| contact_completion | 741.066907 | 741.066907 | 0.000000 | 4008.626655 | 2537 |
| orientation_penalty | -7.982771 | 7.982771 | -185.926970 | -0.043910 | 2537 |
| soft_landing_speed_penalty | -31.523053 | 31.523053 | -186.546389 | -0.002108 | 2537 |
| total_reward | 705.514169 | 737.623504 | -213.948533 | 3931.059058 | 2537 |
