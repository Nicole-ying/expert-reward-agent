# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_penalty | -0.005085 | 0.005085 | 1.000000 | -0.005085 | 0.005085 | -0.225815 | -0.000000 | 1003520 |
| component.contact_reward | 2.746771 | 2.746771 | 0.063566 | 43.211162 | 43.211162 | 0.000000 | 80.000000 | 1003520 |
| component.proximity_improvement | 0.226273 | 0.251602 | 0.999994 | 0.226274 | 0.251603 | -3.107427 | 3.135874 | 1003520 |
| component.total_reward | 1.424129 | 1.459079 | 1.000000 | 1.424129 | 1.459079 | -2.240233 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.071128 | 0.071128 | 0.999952 | -0.071132 | 0.071132 | -0.341654 | -0.000000 | 1003520 |
| generated_reward | 1.424129 | 1.459079 | 1.000000 | 1.424129 | 1.459079 | -2.240233 | 20.000000 | 1003520 |
| original_env_reward | -0.661228 | 3.321573 | 1.000000 | -0.661228 | 3.321573 | -100.000000 | 135.664741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_penalty | -0.696361 | 0.696361 | -14.662724 | -0.017235 | 7320 |
| contact_reward | 375.885246 | 375.885246 | 0.000000 | 7200.000000 | 7320 |
| proximity_improvement | 31.009826 | 31.010379 | -2.024858 | 46.799815 | 7320 |
| total_reward | 194.913306 | 194.952530 | -17.415831 | 3308.018506 | 7320 |
| velocity_penalty | -9.745039 | 9.745039 | -36.426706 | -6.343836 | 7320 |
