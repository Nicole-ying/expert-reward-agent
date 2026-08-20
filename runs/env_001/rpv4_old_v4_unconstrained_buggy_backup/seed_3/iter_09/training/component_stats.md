# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 19.184131 | 19.184131 | 0.327401 | 58.595292 | 58.595292 | 0.000000 | 503.000000 | 1003520 |
| component.goal_proximity | 0.008128 | 0.009190 | 0.999908 | 0.008129 | 0.009191 | -0.059486 | 0.075061 | 1003520 |
| component.orientation_penalty | -0.005663 | 0.005663 | 1.000000 | -0.005663 | 0.005663 | -0.326467 | -0.000000 | 1003520 |
| component.total_reward | 1.564990 | 1.634494 | 1.000000 | 1.564990 | 1.634494 | -1.250237 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.039104 | 0.039104 | 0.999701 | -0.039115 | 0.039115 | -1.114021 | -0.000000 | 1003520 |
| generated_reward | 1.564990 | 1.634494 | 1.000000 | 1.564990 | 1.634494 | -1.250237 | 20.000000 | 1003520 |
| original_env_reward | -0.122912 | 2.923704 | 1.000000 | -0.122912 | 2.923704 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 5298.254897 | 5298.254897 | 0.000000 | 44156.000000 | 3625 |
| goal_proximity | 2.248340 | 2.248340 | 0.117863 | 2.833431 | 3625 |
| orientation_penalty | -1.563816 | 1.563816 | -20.211758 | -0.032382 | 3625 |
| total_reward | 432.430863 | 440.042841 | -47.714649 | 3121.707707 | 3625 |
| velocity_penalty | -10.814817 | 10.814817 | -43.495089 | -3.079593 | 3625 |
