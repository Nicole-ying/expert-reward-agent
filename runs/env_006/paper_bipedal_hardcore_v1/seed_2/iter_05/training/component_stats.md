# Reward Component Training Statistics

- steps_seen: 1120000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.018389 | 0.018389 | 1.000000 | -0.018389 | 0.018389 | -0.040000 | -0.000034 | 1120000 |
| component.forward_reward | 0.183384 | 0.183384 | 0.894819 | 0.204940 | 0.204940 | 0.000000 | 0.733819 | 1120000 |
| component.hinge_penalty | -0.000855 | 0.000855 | 0.025137 | -0.034029 | 0.034029 | -1.491960 | -0.000000 | 1120000 |
| component.total_reward | 0.164140 | 0.171488 | 1.000000 | 0.164140 | 0.171488 | -1.508244 | 0.718778 | 1120000 |
| generated_reward | 0.164140 | 0.171488 | 1.000000 | 0.164140 | 0.171488 | -1.508244 | 0.718778 | 1120000 |
| original_env_reward | -0.562283 | 0.798060 | 1.000000 | -0.562283 | 0.798060 | -100.000000 | 0.836407 | 1120000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -2.836443 | 2.836443 | -33.603633 | -0.635593 | 7254 |
| forward_reward | 28.274319 | 28.274319 | 0.000000 | 196.818776 | 7254 |
| hinge_penalty | -0.132068 | 0.132068 | -24.152123 | 0.000000 | 7254 |
| total_reward | 25.305808 | 25.638026 | -34.205242 | 183.767924 | 7254 |
