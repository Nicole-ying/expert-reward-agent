# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_improvement | 0.004673 | 0.005538 | 0.998713 | 0.004679 | 0.005545 | -0.133464 | 0.162840 | 1003520 |
| component.contact_reward | 1.584211 | 1.584211 | 0.528070 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 1003520 |
| component.soft_landing_velocity | -0.016199 | 0.016199 | 0.999519 | -0.016207 | 0.016207 | -0.746238 | -0.000000 | 1003520 |
| component.total_reward | 1.567231 | 1.597718 | 1.000000 | 1.567231 | 1.597718 | -5.252032 | 3.007585 | 1003520 |
| component.upright_stabilization | -0.005454 | 0.005454 | 0.999896 | -0.005454 | 0.005454 | -5.038380 | -0.000000 | 1003520 |
| generated_reward | 1.567231 | 1.597718 | 1.000000 | 1.567231 | 1.597718 | -5.252032 | 3.007585 | 1003520 |
| original_env_reward | -0.015864 | 1.469755 | 1.000000 | -0.015864 | 1.469755 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_improvement | 1.864861 | 1.866245 | -0.311895 | 2.922879 | 2510 |
| contact_reward | 632.396414 | 632.396414 | 0.000000 | 2619.000000 | 2510 |
| soft_landing_velocity | -6.470341 | 6.470341 | -19.595356 | -1.893983 | 2510 |
| total_reward | 625.611515 | 632.384041 | -39.382447 | 2613.837931 | 2510 |
| upright_stabilization | -2.179419 | 2.179419 | -29.030943 | -0.016934 | 2510 |
