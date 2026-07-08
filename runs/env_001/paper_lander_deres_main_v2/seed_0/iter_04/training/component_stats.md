# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.progress_delta | -0.001125 | 0.005831 | 1.000000 | -0.001125 | 0.005831 | -0.051037 | 0.060387 | 1003520 |
| component.soft_landing_proxy | 0.000064 | 0.000064 | 0.000771 | 0.082905 | 0.082905 | 0.000000 | 0.485593 | 1003520 |
| component.stability_penalty | -0.004788 | 0.004788 | 0.308173 | -0.015536 | 0.015536 | -0.729799 | -0.000000 | 1003520 |
| component.total_reward | -0.005849 | 0.009108 | 1.000000 | -0.005849 | 0.009108 | -0.682371 | 0.478278 | 1003520 |
| generated_reward | -0.005849 | 0.009108 | 1.000000 | -0.005849 | 0.009108 | -0.682371 | 0.478278 | 1003520 |
| original_env_reward | -4.029209 | 4.383759 | 1.000000 | -4.029209 | 4.383759 | -100.000000 | 152.518886 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| progress_delta | -0.144724 | 0.260748 | -6.161339 | 1.412644 | 7794 |
| soft_landing_proxy | 0.008233 | 0.008233 | 0.000000 | 24.960482 | 7794 |
| stability_penalty | -0.616433 | 0.616433 | -18.365666 | -0.008350 | 7794 |
| total_reward | -0.752924 | 0.755017 | -17.356452 | 8.156460 | 7794 |
