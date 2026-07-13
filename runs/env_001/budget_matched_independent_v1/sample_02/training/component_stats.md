# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.distance_progress | 0.003437 | 0.003787 | 0.998956 | 0.003440 | 0.003791 | -0.038677 | 0.040607 | 1003520 |
| component.orientation_penalty | -0.003150 | 0.003150 | 0.999897 | -0.003150 | 0.003150 | -2.684093 | -0.000000 | 1003520 |
| component.soft_landing_bonus | 0.260871 | 0.260871 | 0.558838 | 0.466809 | 0.466809 | 0.000000 | 0.500000 | 1003520 |
| component.total_reward | 0.243086 | 0.276735 | 1.000000 | 0.243086 | 0.276735 | -2.771926 | 0.499999 | 1003520 |
| component.velocity_damping | -0.018072 | 0.018072 | 0.999931 | -0.018073 | 0.018073 | -0.231130 | -0.000000 | 1003520 |
| generated_reward | 0.243086 | 0.276735 | 1.000000 | 0.243086 | 0.276735 | -2.771926 | 0.499999 | 1003520 |
| original_env_reward | -0.029158 | 1.378554 | 1.000000 | -0.029158 | 1.378554 | -100.000000 | 132.235189 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| distance_progress | 1.143751 | 1.144806 | -0.536427 | 1.420915 | 3011 |
| orientation_penalty | -1.049466 | 1.049466 | -19.625906 | -0.004865 | 3011 |
| soft_landing_bonus | 86.869843 | 86.869843 | 0.000000 | 432.117372 | 3011 |
| total_reward | 80.947730 | 88.757914 | -24.836590 | 426.539377 | 3011 |
| velocity_damping | -6.016397 | 6.016397 | -15.913520 | -3.989951 | 3011 |
