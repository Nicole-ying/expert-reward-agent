# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_progress | -0.972942 | 0.972942 | 1.000000 | -0.972942 | 0.972942 | -1.735195 | -0.001527 | 1003520 |
| component.total_reward | -1.056269 | 1.068421 | 1.000000 | -1.056269 | 1.068421 | -15.314019 | 5.769123 | 1003520 |
| component.touchdown_proxy | 0.006206 | 0.006206 | 0.001899 | 3.267538 | 3.267538 | 0.000000 | 5.811146 | 1003520 |
| component.upright_stabilization | -0.013803 | 0.013803 | 0.999979 | -0.013804 | 0.013804 | -15.077470 | -0.000000 | 1003520 |
| component.velocity_damping | -0.075730 | 0.075730 | 0.999882 | -0.075739 | 0.075739 | -0.924717 | -0.000000 | 1003520 |
| generated_reward | -1.056269 | 1.068421 | 1.000000 | -1.056269 | 1.068421 | -15.314019 | 5.769123 | 1003520 |
| original_env_reward | -1.607023 | 2.424898 | 1.000000 | -1.607023 | 2.424898 | -100.000000 | 131.538913 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_progress | -68.330297 | 68.330297 | -193.520711 | -40.553069 | 14286 |
| total_reward | -74.183032 | 74.183032 | -348.390869 | -41.942406 | 14286 |
| touchdown_proxy | 0.435946 | 0.435946 | 0.000000 | 5.811146 | 14286 |
| upright_stabilization | -0.969602 | 0.969602 | -219.033300 | -0.000572 | 14286 |
| velocity_damping | -5.319079 | 5.319079 | -8.883654 | -3.362794 | 14286 |
