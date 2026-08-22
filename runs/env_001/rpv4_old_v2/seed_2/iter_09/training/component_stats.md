# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_distance_progress | 0.014661 | 0.016338 | 0.998812 | 0.014678 | 0.016358 | -0.172501 | 0.185942 | 1003520 |
| component.landing_contact_success | 1.504783 | 1.504783 | 0.547689 | 2.747512 | 2.747512 | 0.000000 | 4.989124 | 1003520 |
| component.orientation_penalty | -0.010012 | 0.010012 | 0.999783 | -0.010014 | 0.010014 | -11.709076 | -0.000000 | 1003520 |
| component.soft_landing | -0.003546 | 0.003546 | 0.999199 | -0.003549 | 0.003549 | -2.446883 | -0.000000 | 1003520 |
| component.total_reward | 1.505886 | 1.520932 | 1.000000 | 1.505886 | 1.520932 | -11.792232 | 4.989124 | 1003520 |
| generated_reward | 1.505886 | 1.520932 | 1.000000 | 1.505886 | 1.520932 | -11.792232 | 4.989124 | 1003520 |
| original_env_reward | -0.001783 | 1.426030 | 1.000000 | -0.001783 | 1.426030 | -100.000000 | 121.799245 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_distance_progress | 5.531489 | 5.531946 | -0.607128 | 7.098593 | 2655 |
| landing_contact_success | 567.230877 | 567.230877 | 0.000000 | 4012.881177 | 2655 |
| orientation_penalty | -3.782702 | 3.782702 | -206.995912 | -0.006307 | 2655 |
| soft_landing | -1.338355 | 1.338355 | -17.420520 | -0.000021 | 2655 |
| total_reward | 567.641310 | 570.641649 | -205.678525 | 4016.293304 | 2655 |
