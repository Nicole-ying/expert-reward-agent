# Reward Component Training Statistics

- steps_seen: 1680000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.017586 | 0.017586 | 1.000000 | -0.017586 | 0.017586 | -0.040000 | -0.000008 | 1680000 |
| component.contact_transition_reward | 0.097118 | 0.099077 | 0.985876 | 0.098509 | 0.100497 | -0.200000 | 0.100000 | 1680000 |
| component.forward_reward_gated | 0.239089 | 0.239089 | 0.881114 | 0.271349 | 0.271349 | 0.000000 | 1.314130 | 1680000 |
| component.total_reward | 0.318621 | 0.319713 | 1.000000 | 0.318621 | 0.319713 | -0.240000 | 1.400415 | 1680000 |
| generated_reward | 0.318621 | 0.319713 | 1.000000 | 0.318621 | 0.319713 | -0.240000 | 1.400415 | 1680000 |
| original_env_reward | -0.264222 | 0.464758 | 1.000000 | -0.264222 | 0.464758 | -100.000000 | 0.770061 | 1680000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -5.251168 | 5.251168 | -33.595450 | -0.670365 | 5620 |
| contact_transition_reward | 28.993203 | 28.993203 | 2.400000 | 159.700000 | 5620 |
| forward_reward_gated | 71.373992 | 71.373992 | 0.003332 | 655.220089 | 5620 |
| total_reward | 95.116028 | 95.116028 | 1.793322 | 785.890092 | 5620 |
