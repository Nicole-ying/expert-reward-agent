# Reward Component Training Statistics

- steps_seen: 1280000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_transition_reward | 0.066757 | 0.066757 | 0.996191 | 0.067012 | 0.067012 | 0.000000 | 0.275150 | 1280000 |
| component.energy_penalty | -0.010544 | 0.010544 | 1.000000 | -0.010544 | 0.010544 | -0.020000 | -0.000021 | 1280000 |
| component.forward_reward | 0.150442 | 0.150442 | 0.864298 | 0.174062 | 0.174062 | 0.000000 | 0.790603 | 1280000 |
| component.gated_forward | 0.144142 | 0.144142 | 0.858158 | 0.167966 | 0.167966 | 0.000000 | 0.771275 | 1280000 |
| component.stability_gate | 0.950874 | 0.950874 | 0.989039 | 0.961412 | 0.961412 | 0.000000 | 1.000000 | 1280000 |
| component.total_reward | 0.200355 | 0.201010 | 1.000000 | 0.200355 | 0.201010 | -0.020000 | 0.894789 | 1280000 |
| generated_reward | 0.200355 | 0.201010 | 1.000000 | 0.200355 | 0.201010 | -0.020000 | 0.894789 | 1280000 |
| original_env_reward | -0.325210 | 0.476544 | 1.000000 | -0.325210 | 0.476544 | -100.000000 | 0.804664 | 1280000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_transition_reward | 18.653330 | 18.653330 | 1.534454 | 126.051615 | 4576 |
| energy_penalty | -2.946366 | 2.946366 | -18.385158 | -0.343485 | 4576 |
| forward_reward | 42.024319 | 42.024319 | 0.000850 | 304.765067 | 4576 |
| gated_forward | 40.263210 | 40.263210 | 0.000850 | 302.450560 | 4576 |
| stability_gate | 265.704853 | 265.704853 | 9.675888 | 1599.806978 | 4576 |
| total_reward | 55.970174 | 55.970174 | 1.236085 | 375.209742 | 4576 |
