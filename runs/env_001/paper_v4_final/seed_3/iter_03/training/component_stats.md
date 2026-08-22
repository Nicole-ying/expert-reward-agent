# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.019207 | 0.019207 | 0.994585 | -0.019311 | 0.019311 | -7.176230 | -0.000000 | 1003520 |
| component.landing_bonus | 0.016690 | 0.016690 | 0.007452 | 2.239772 | 2.239772 | 0.000000 | 2.999990 | 1003520 |
| component.lateral_drift_penalty | -0.038974 | 0.038974 | 0.996705 | -0.039102 | 0.039102 | -0.711738 | -0.000000 | 1003520 |
| component.progress_gated | 0.145784 | 0.154377 | 0.999997 | 0.145785 | 0.154377 | -0.357604 | 0.397776 | 1003520 |
| component.total_reward | 0.104294 | 0.165312 | 0.999997 | 0.104294 | 0.165312 | -7.206820 | 2.993171 | 1003520 |
| generated_reward | 0.104294 | 0.165312 | 0.999997 | 0.104294 | 0.165312 | -7.206820 | 2.993171 | 1003520 |
| original_env_reward | -1.313193 | 2.540621 | 1.000000 | -1.313193 | 2.540621 | -100.000000 | 133.260892 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -1.373108 | 1.373108 | -24.405016 | -0.002309 | 14037 |
| landing_bonus | 1.193205 | 1.193205 | 0.000000 | 1240.116363 | 14037 |
| lateral_drift_penalty | -2.785545 | 2.785545 | -25.375667 | -0.000682 | 14037 |
| progress_gated | 10.421627 | 10.422378 | -1.303485 | 13.195382 | 14037 |
| total_reward | 7.456179 | 8.150170 | -35.047226 | 1229.851965 | 14037 |
