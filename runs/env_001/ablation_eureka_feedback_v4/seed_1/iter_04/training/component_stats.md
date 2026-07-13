# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 0.029800 | 0.029800 | 0.029800 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1003520 |
| component.orientation_penalty | -0.014195 | 0.014195 | 0.999988 | -0.014195 | 0.014195 | -7.188108 | -0.000000 | 1003520 |
| component.proximity_reward | -0.075610 | 0.075610 | 1.000000 | -0.075610 | 0.075610 | -0.281238 | -0.000000 | 1003520 |
| component.total_reward | -0.078607 | 0.134006 | 1.000000 | -0.078607 | 0.134006 | -7.244332 | 1.000000 | 1003520 |
| component.velocity_penalty | -0.018602 | 0.018602 | 0.998813 | -0.018624 | 0.018624 | -0.324792 | -0.000000 | 1003520 |
| generated_reward | -0.078607 | 0.134006 | 1.000000 | -0.078607 | 0.134006 | -7.244332 | 1.000000 | 1003520 |
| original_env_reward | -0.378866 | 3.275301 | 1.000000 | -0.378866 | 3.275301 | -100.000000 | 141.821033 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 3.209079 | 3.209079 | 0.000000 | 107.000000 | 9274 |
| orientation_penalty | -1.535349 | 1.535349 | -122.771572 | -0.005166 | 9274 |
| proximity_reward | -8.177365 | 8.177365 | -57.340134 | -4.141846 | 9274 |
| total_reward | -8.516110 | 12.392563 | -146.477134 | 98.992233 | 9274 |
| velocity_penalty | -2.012475 | 2.012475 | -3.778649 | -0.966186 | 9274 |
