# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.fuel_penalty | -0.013256 | 0.013256 | 0.662808 | -0.020000 | 0.020000 | -0.020000 | 0.000000 | 1003520 |
| component.progress_reward | 0.005752 | 0.006226 | 0.999110 | 0.005757 | 0.006232 | -0.062055 | 0.075876 | 1003520 |
| component.safe_contact_bonus | 0.224542 | 0.224542 | 0.609013 | 0.368698 | 0.368698 | 0.000000 | 0.499724 | 1003520 |
| component.stability_penalty | -0.021393 | 0.021393 | 1.000000 | -0.021393 | 0.021393 | -5.322875 | -0.000000 | 1003520 |
| component.total_reward | 0.195645 | 0.237559 | 1.000000 | 0.195645 | 0.237559 | -5.331846 | 0.499719 | 1003520 |
| generated_reward | 0.195645 | 0.237559 | 1.000000 | 0.195645 | 0.237559 | -5.331846 | 0.499719 | 1003520 |
| original_env_reward | -0.008060 | 1.239595 | 1.000000 | -0.008060 | 1.239595 | -100.000000 | 144.013810 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| fuel_penalty | -5.472434 | 5.472434 | -17.600000 | -0.340000 | 2424 |
| progress_reward | 2.377143 | 2.377513 | -0.448384 | 2.840299 | 2424 |
| safe_contact_bonus | 92.635395 | 92.635395 | 0.000000 | 400.270855 | 2424 |
| stability_penalty | -8.849838 | 8.849838 | -59.070100 | -2.427063 | 2424 |
| total_reward | 80.690267 | 93.325494 | -57.830245 | 387.672299 | 2424 |
