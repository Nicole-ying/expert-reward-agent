# Reward Component Training Statistics

- steps_seen: 1360000
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.energy_penalty | -0.017614 | 0.017614 | 1.000000 | -0.017614 | 0.017614 | -0.040000 | -0.000018 | 1360000 |
| component.forward_reward | 0.169572 | 0.169572 | 0.883768 | 0.191874 | 0.191874 | 0.000000 | 0.660968 | 1360000 |
| component.hinge_penalty | -0.004169 | 0.004169 | 0.063567 | -0.065581 | 0.065581 | -1.147262 | -0.000000 | 1360000 |
| component.total_reward | 0.147789 | 0.158875 | 1.000000 | 0.147789 | 0.158875 | -1.170350 | 0.658126 | 1360000 |
| generated_reward | 0.147789 | 0.158875 | 1.000000 | 0.147789 | 0.158875 | -1.170350 | 0.658126 | 1360000 |
| original_env_reward | -0.480255 | 0.698064 | 1.000000 | -0.480255 | 0.698064 | -100.000000 | 0.754033 | 1360000 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| energy_penalty | -3.126606 | 3.126606 | -33.603633 | -0.669977 | 7657 |
| forward_reward | 30.089475 | 30.089475 | 0.000000 | 263.140938 | 7657 |
| hinge_penalty | -0.740429 | 0.740429 | -90.691332 | 0.000000 | 7657 |
| total_reward | 26.222440 | 26.965803 | -101.293642 | 247.739604 | 7657 |
