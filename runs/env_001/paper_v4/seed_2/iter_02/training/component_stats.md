# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_completion | 1.405609 | 1.405609 | 1.000000 | 1.405609 | 1.405609 | 0.107438 | 2.000000 | 1003520 |
| component.position_proximity | 0.689263 | 0.689263 | 1.000000 | 0.689263 | 0.689263 | 0.145072 | 0.999965 | 1003520 |
| component.soft_landing_velocity | -0.003626 | 0.003626 | 0.999853 | -0.003627 | 0.003627 | -1.307995 | -0.000000 | 1003520 |
| component.stable_orientation | -0.030404 | 0.030404 | 0.999931 | -0.030406 | 0.030406 | -5.732079 | -0.000000 | 1003520 |
| component.total_reward | 2.060841 | 2.064146 | 1.000000 | 2.060841 | 2.064146 | -4.802255 | 2.999885 | 1003520 |
| generated_reward | 2.060841 | 2.064146 | 1.000000 | 2.060841 | 2.064146 | -4.802255 | 2.999885 | 1003520 |
| original_env_reward | -0.159432 | 2.410434 | 1.000000 | -0.159432 | 2.410434 | -100.000000 | 149.733164 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_completion | 710.419272 | 710.419272 | 49.133545 | 1793.174717 | 1983 |
| position_proximity | 348.355007 | 348.355007 | 30.810386 | 914.805265 | 1983 |
| soft_landing_velocity | -1.831910 | 1.831910 | -15.475383 | -0.244083 | 1983 |
| stable_orientation | -15.375581 | 15.375581 | -203.608310 | -0.023118 | 1983 |
| total_reward | 1041.566788 | 1041.566788 | 49.107383 | 2689.902508 | 1983 |
