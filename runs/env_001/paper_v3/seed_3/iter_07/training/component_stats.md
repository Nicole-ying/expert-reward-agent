# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_improvement | 0.030294 | 0.030294 | 0.679354 | 0.044593 | 0.044593 | 0.000000 | 0.350745 | 1003520 |
| component.contact_reward | 1.535087 | 1.535087 | 0.511696 | 3.000000 | 3.000000 | 0.000000 | 3.000000 | 1003520 |
| component.soft_landing_velocity | -0.007715 | 0.007715 | 0.999391 | -0.007720 | 0.007720 | -0.395516 | -0.000000 | 1003520 |
| component.total_reward | 1.552883 | 1.558534 | 1.000000 | 1.552883 | 1.558534 | -7.225645 | 3.031392 | 1003520 |
| component.upright_stabilization | -0.004783 | 0.004783 | 0.999938 | -0.004783 | 0.004783 | -7.127302 | -0.000000 | 1003520 |
| generated_reward | 1.552883 | 1.558534 | 1.000000 | 1.552883 | 1.558534 | -7.225645 | 3.031392 | 1003520 |
| original_env_reward | 0.001729 | 1.536707 | 1.000000 | 0.001729 | 1.536707 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_improvement | 11.975199 | 11.975199 | 1.928513 | 16.105093 | 2535 |
| contact_reward | 606.204734 | 606.204734 | 0.000000 | 2583.000000 | 2535 |
| soft_landing_velocity | -3.052306 | 3.052306 | -13.007599 | -0.935101 | 2535 |
| total_reward | 613.234976 | 613.745899 | -29.888180 | 2593.908635 | 2535 |
| upright_stabilization | -1.892651 | 1.892651 | -31.674870 | -0.011128 | 2535 |
