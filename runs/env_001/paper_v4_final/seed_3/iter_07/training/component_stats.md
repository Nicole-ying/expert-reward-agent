# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angvel_penalty | -0.002024 | 0.002024 | 0.913958 | -0.002215 | 0.002215 | -2.458234 | -0.000000 | 1003520 |
| component.contact_landing_reward | 2.166358 | 2.166358 | 0.538741 | 4.021151 | 4.021151 | 0.000000 | 4.998754 | 1003520 |
| component.lateral_pos_penalty | -0.008180 | 0.008180 | 0.999985 | -0.008180 | 0.008180 | -0.082636 | -0.000000 | 1003520 |
| component.progress_gated | 0.120255 | 0.138116 | 0.999169 | 0.120355 | 0.138231 | -0.796373 | 1.034925 | 1003520 |
| component.total_reward | 2.276408 | 2.290999 | 1.000000 | 2.276408 | 2.290999 | -2.303109 | 5.116702 | 1003520 |
| generated_reward | 2.276408 | 2.290999 | 1.000000 | 2.276408 | 2.290999 | -2.303109 | 5.116702 | 1003520 |
| original_env_reward | -0.100375 | 1.825470 | 1.000000 | -0.100375 | 1.825470 | -100.000000 | 128.588050 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angvel_penalty | -0.484640 | 0.484640 | -8.570416 | -0.002587 | 4189 |
| contact_landing_reward | 516.942459 | 516.942459 | 0.000000 | 4533.433452 | 4189 |
| lateral_pos_penalty | -1.957460 | 1.957460 | -52.032628 | -0.000005 | 4189 |
| progress_gated | 28.781032 | 28.784748 | -2.961814 | 41.171199 | 4189 |
| total_reward | 543.281390 | 543.311367 | -15.751363 | 4568.661980 | 4189 |
