# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.048195 | 0.048195 | 0.999977 | -0.048196 | 0.048196 | -9.899705 | -0.000000 | 1003520 |
| component.approach_progress | 0.008206 | 0.009340 | 0.998814 | 0.008215 | 0.009351 | -0.063269 | 0.091449 | 1003520 |
| component.landing_quality | 1.609914 | 1.609914 | 0.405524 | 3.969965 | 3.969965 | 0.000000 | 4.998271 | 1003520 |
| component.total_reward | 0.487131 | 2.689838 | 1.000000 | 0.487131 | 2.689838 | -16.208103 | 4.998271 | 1003520 |
| component.velocity_penalty | -1.082794 | 1.082794 | 0.508242 | -2.130470 | 2.130470 | -13.398504 | -0.000000 | 1003520 |
| generated_reward | 0.487131 | 2.689838 | 1.000000 | 0.487131 | 2.689838 | -16.208103 | 4.998271 | 1003520 |
| original_env_reward | -0.343634 | 1.913838 | 1.000000 | -0.343634 | 1.913838 | -100.000000 | 140.758745 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -12.426111 | 12.426111 | -146.120969 | -0.031253 | 3891 |
| approach_progress | 2.113729 | 2.122208 | -1.060417 | 2.842348 | 3891 |
| landing_quality | 413.894111 | 413.894111 | 0.000000 | 4097.326372 | 3891 |
| total_reward | 124.511676 | 638.299628 | -663.034425 | 3849.574216 | 3891 |
| velocity_penalty | -279.070053 | 279.070053 | -684.517036 | -17.030197 | 3891 |
