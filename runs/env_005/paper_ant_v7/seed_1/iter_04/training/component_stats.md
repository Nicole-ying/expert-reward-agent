# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_penalty | -0.043764 | 0.043764 | 1.000000 | -0.043764 | 0.043764 | -0.080000 | -0.001014 | 1001472 |
| component.forward_gated | 0.210570 | 0.324436 | 0.411576 | 0.511617 | 0.788276 | -3.724648 | 5.916084 | 1001472 |
| component.lateral_penalty | -0.055480 | 0.055480 | 0.411576 | -0.134799 | 0.134799 | -4.347332 | -0.000000 | 1001472 |
| component.total_reward | 0.111326 | 0.328351 | 1.000000 | 0.111326 | 0.328351 | -4.952458 | 5.797376 | 1001472 |
| generated_reward | 0.111326 | 0.328351 | 1.000000 | 0.111326 | 0.328351 | -4.952458 | 5.797376 | 1001472 |
| original_env_reward | -0.958536 | 1.123751 | 1.000000 | -0.958536 | 1.123751 | -5.803635 | 5.330887 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_penalty | -8.489428 | 8.489428 | -54.158478 | -0.277436 | 5159 |
| forward_gated | 40.871592 | 43.682828 | -210.459606 | 853.874073 | 5159 |
| lateral_penalty | -10.768811 | 10.768811 | -125.076549 | -0.043775 | 5159 |
| total_reward | 21.613353 | 34.153345 | -305.096105 | 680.312155 | 5159 |
