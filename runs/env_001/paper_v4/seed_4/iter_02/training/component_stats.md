# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.015816 | 0.015816 | 0.790777 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.progress_reward | 0.005549 | 0.006311 | 0.999962 | 0.005549 | 0.006311 | -0.053088 | 0.069663 | 1003520 |
| component.safe_contact_bonus | 0.489991 | 0.489991 | 0.069947 | 7.005195 | 7.005195 | 0.000000 | 9.990643 | 1003520 |
| component.stability_penalty | -0.020919 | 0.020919 | 1.000000 | -0.020919 | 0.020919 | -4.971599 | -0.000000 | 1003520 |
| component.total_reward | 0.458806 | 0.517983 | 1.000000 | 0.458806 | 0.517983 | -4.972484 | 9.990759 | 1003520 |
| generated_reward | 0.458806 | 0.517983 | 1.000000 | 0.458806 | 0.517983 | -4.972484 | 9.990759 | 1003520 |
| original_env_reward | -0.013423 | 3.853974 | 1.000000 | -0.013423 | 3.853974 | -100.000000 | 137.864768 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -6.800507 | 6.800507 | -18.900000 | -0.740000 | 2329 |
| progress_reward | 2.386335 | 2.386720 | -0.448384 | 2.841501 | 2329 |
| safe_contact_bonus | 210.638589 | 210.638589 | 0.000000 | 1075.356661 | 2329 |
| stability_penalty | -9.002891 | 9.002891 | -43.957635 | -3.065161 | 2329 |
| total_reward | 197.221527 | 205.810789 | -44.602168 | 1056.431766 | 2329 |
