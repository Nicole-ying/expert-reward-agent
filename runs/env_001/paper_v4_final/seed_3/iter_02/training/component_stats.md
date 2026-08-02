# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.019678 | 0.019678 | 0.994528 | -0.019787 | 0.019787 | -6.455377 | -0.000000 | 1003520 |
| component.landing_bonus | 0.015592 | 0.015592 | 0.006485 | 2.404324 | 2.404324 | 0.000000 | 2.999978 | 1003520 |
| component.lateral_drift_penalty | -0.039092 | 0.039092 | 0.996757 | -0.039219 | 0.039219 | -0.781171 | -0.000000 | 1003520 |
| component.progress_gated | 0.143865 | 0.152442 | 0.999996 | 0.143866 | 0.152442 | -0.379302 | 0.403450 | 1003520 |
| component.total_reward | 0.100688 | 0.163474 | 0.999997 | 0.100688 | 0.163474 | -6.456231 | 2.993838 | 1003520 |
| generated_reward | 0.100688 | 0.163474 | 0.999997 | 0.100688 | 0.163474 | -6.456231 | 2.993838 | 1003520 |
| original_env_reward | -1.369624 | 2.527288 | 1.000000 | -1.369624 | 2.527288 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -1.406622 | 1.406622 | -36.968918 | -0.000594 | 14039 |
| landing_bonus | 1.114563 | 1.114563 | 0.000000 | 1069.925055 | 14039 |
| lateral_drift_penalty | -2.793618 | 2.793618 | -26.193814 | -0.000350 | 14039 |
| progress_gated | 10.282543 | 10.283059 | -1.162946 | 13.209740 | 14039 |
| total_reward | 7.196866 | 8.053721 | -58.018400 | 1059.588742 | 14039 |
