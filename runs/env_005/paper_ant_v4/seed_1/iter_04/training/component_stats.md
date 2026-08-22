# Reward Component Training Statistics

- steps_seen: 1015808
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 2.747451 | 17.427841 | 1.000000 | 2.747451 | 17.427841 | -48.461668 | 49.918703 | 1015808 |
| component.height_penalty | -1.058947 | 1.058947 | 0.491270 | -2.155529 | 2.155529 | -5.999029 | 0.000000 | 1015808 |
| component.lateral_drift_penalty | -0.048887 | 0.048887 | 0.999995 | -0.048887 | 0.048887 | -2.498982 | -0.000000 | 1015808 |
| component.total_reward | 0.685775 | 14.071643 | 1.000000 | 0.685775 | 14.071643 | -20.000000 | 20.000000 | 1015808 |
| component.upright_stability | -0.145382 | 0.936129 | 1.000000 | -0.145382 | 0.936129 | -1.957961 | 0.999981 | 1015808 |
| generated_reward | 0.685775 | 14.071643 | 1.000000 | 0.685775 | 14.071643 | -20.000000 | 20.000000 | 1015808 |
| original_env_reward | -0.927678 | 1.082879 | 1.000000 | -0.927678 | 1.082879 | -5.604307 | 4.209287 | 1015808 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 362.716603 | 591.774624 | -3237.131740 | 5756.706122 | 7671 |
| height_penalty | -139.219672 | 139.219672 | -1931.073642 | 0.000000 | 7671 |
| lateral_drift_penalty | -6.454861 | 6.454861 | -58.701256 | -0.029247 | 7671 |
| total_reward | 91.265662 | 478.659625 | -4126.710554 | 3907.496896 | 7671 |
| upright_stability | -18.780758 | 114.467609 | -935.188598 | 581.322050 | 7671 |
