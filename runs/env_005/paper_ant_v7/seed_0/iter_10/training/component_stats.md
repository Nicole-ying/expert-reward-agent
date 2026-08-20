# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.200491 | 0.200491 | 1.000000 | -0.200491 | 0.200491 | -0.400000 | -0.005636 | 1001472 |
| component.forward_reward | 0.909830 | 1.192091 | 1.000000 | 0.909830 | 1.192091 | -4.434719 | 19.331264 | 1001472 |
| component.height_boundary_penalty | -0.000755 | 0.000755 | 0.007019 | -0.107601 | 0.107601 | -0.299987 | -0.000000 | 1001472 |
| component.height_reward | 0.193645 | 0.193645 | 1.000000 | 0.193645 | 0.193645 | 0.000002 | 0.400000 | 1001472 |
| component.lateral_velocity_penalty | -0.028330 | 0.028330 | 0.583439 | -0.048557 | 0.048557 | -0.466157 | -0.000000 | 1001472 |
| component.total_reward | 0.873900 | 1.220593 | 1.000000 | 0.873900 | 1.220593 | -4.271265 | 19.515393 | 1001472 |
| generated_reward | 0.873900 | 1.220593 | 1.000000 | 0.873900 | 1.220593 | -4.271265 | 19.515393 | 1001472 |
| original_env_reward | -0.561073 | 0.947296 | 1.000000 | -0.561073 | 0.947296 | -5.625363 | 4.925436 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -55.649002 | 55.649002 | -225.985288 | -1.496412 | 3608 |
| forward_reward | 252.537996 | 254.317146 | -82.448864 | 2954.233762 | 3608 |
| height_boundary_penalty | -0.209625 | 0.209625 | -2.994642 | 0.000000 | 3608 |
| height_reward | 53.748730 | 53.748730 | 1.825516 | 346.849839 | 3608 |
| lateral_velocity_penalty | -7.863452 | 7.863452 | -52.617391 | -0.006014 | 3608 |
| total_reward | 242.564646 | 247.240480 | -162.165800 | 3017.491544 | 3608 |
