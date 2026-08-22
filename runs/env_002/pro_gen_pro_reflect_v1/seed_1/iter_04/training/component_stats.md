# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_efficiency | -0.033936 | 0.033936 | 1.000000 | -0.033936 | 0.033936 | -0.080000 | -0.000040 | 1015808 |
| component.stability_penalty | -0.000152 | 0.000152 | 0.999922 | -0.000152 | 0.000152 | -0.003938 | -0.000000 | 1015808 |
| component.total_reward | 0.270346 | 0.281504 | 1.000000 | 0.270346 | 0.281504 | -0.416717 | 0.851940 | 1015808 |
| component.velocity_gated | 0.304434 | 0.306759 | 0.991890 | 0.306923 | 0.309267 | -0.353197 | 0.860814 | 1015808 |
| generated_reward | 0.270346 | 0.281504 | 1.000000 | 0.270346 | 0.281504 | -0.416717 | 0.851940 | 1015808 |
| original_env_reward | -0.479874 | 0.828975 | 1.000000 | -0.479874 | 0.828975 | -100.000000 | 0.988143 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_efficiency | -5.372990 | 5.372990 | -67.354176 | -1.229477 | 6403 |
| stability_penalty | -0.024045 | 0.024045 | -0.305365 | -0.004057 | 6403 |
| total_reward | 42.734505 | 43.492383 | -90.192703 | 475.611019 | 6403 |
| velocity_gated | 48.131540 | 48.183124 | -24.646869 | 504.874472 | 6403 |
