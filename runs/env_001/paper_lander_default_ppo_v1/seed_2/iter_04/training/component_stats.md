# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.005088 | 0.005891 | 0.985132 | 0.005165 | 0.005980 | -0.079665 | 0.072056 | 1001472 |
| component.landing_approach | 0.208206 | 0.208206 | 0.989091 | 0.210502 | 0.210502 | 0.000000 | 0.299952 | 1001472 |
| component.stability_penalty | -0.002891 | 0.002891 | 1.000000 | -0.002891 | 0.002891 | -0.096639 | -0.000000 | 1001472 |
| component.total_reward | 0.210403 | 0.210619 | 1.000000 | 0.210403 | 0.210619 | -0.141238 | 0.299950 | 1001472 |
| generated_reward | 0.210403 | 0.210619 | 1.000000 | 0.210403 | 0.210619 | -0.141238 | 0.299950 | 1001472 |
| original_env_reward | 0.122133 | 1.094324 | 1.000000 | 0.122133 | 1.094324 | -100.000000 | 100.132269 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 2.307762 | 2.307762 | 0.517289 | 2.836556 | 2207 |
| landing_approach | 94.432038 | 94.432038 | 1.079248 | 278.649490 | 2207 |
| stability_penalty | -1.311314 | 1.311314 | -5.923119 | -0.566180 | 2207 |
| total_reward | 95.428487 | 95.428490 | -0.004232 | 280.681069 | 2207 |
