# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.002816 | 0.002816 | 0.999588 | -0.002817 | 0.002817 | -1.443576 | -0.000000 | 1001472 |
| component.angular_velocity_penalty | -0.001372 | 0.001372 | 0.794302 | -0.001727 | 0.001727 | -3.200233 | -0.000000 | 1001472 |
| component.contact_bonus | 0.597139 | 0.597139 | 0.626409 | 0.953273 | 0.953273 | 0.000000 | 0.999975 | 1001472 |
| component.proximity_reward | 1.722565 | 1.722565 | 1.000000 | 1.722565 | 1.722565 | 0.685204 | 1.999949 | 1001472 |
| component.total_reward | 2.309271 | 2.309397 | 1.000000 | 2.309271 | 2.309397 | -2.424639 | 2.999858 | 1001472 |
| component.velocity_penalty | -0.006245 | 0.006245 | 0.987279 | -0.006326 | 0.006326 | -0.812020 | -0.000000 | 1001472 |
| generated_reward | 2.309271 | 2.309397 | 1.000000 | 2.309271 | 2.309397 | -2.424639 | 2.999858 | 1001472 |
| original_env_reward | 0.127290 | 1.002054 | 1.000000 | 0.127290 | 1.002054 | -100.000000 | 126.796635 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -1.611239 | 1.611239 | -27.104963 | -0.005682 | 1750 |
| angular_velocity_penalty | -0.784986 | 0.784986 | -9.038708 | -0.018043 | 1750 |
| contact_bonus | 341.531878 | 341.531878 | 0.000000 | 885.195939 | 1750 |
| proximity_reward | 985.260271 | 985.260271 | 63.245705 | 1911.120551 | 1750 |
| total_reward | 1320.822971 | 1320.822971 | 51.051549 | 2784.548057 | 1750 |
| velocity_penalty | -3.572953 | 3.572953 | -21.926578 | -1.324102 | 1750 |
