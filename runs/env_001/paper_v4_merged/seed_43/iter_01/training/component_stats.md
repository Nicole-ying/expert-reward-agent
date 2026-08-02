# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.action_cost | -0.001341 | 0.001341 | 0.134083 | -0.010000 | 0.010000 | -0.010000 | -0.000000 | 1003520 |
| component.angle_hinge | -0.001285 | 0.001285 | 0.009473 | -0.135632 | 0.135632 | -1.632757 | -0.000000 | 1003520 |
| component.progress_shaping | 0.015309 | 0.016966 | 1.000000 | 0.015309 | 0.016966 | -0.065920 | 1.013495 | 1003520 |
| component.total_reward | 0.012683 | 0.017748 | 1.000000 | 0.012683 | 0.017748 | -1.594505 | 1.013495 | 1003520 |
| generated_reward | 0.012683 | 0.017748 | 1.000000 | 0.012683 | 0.017748 | -1.594505 | 1.013495 | 1003520 |
| original_env_reward | -1.571001 | 2.346916 | 1.000000 | -1.571001 | 2.346916 | -100.000000 | 134.270033 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| action_cost | -0.094594 | 0.094594 | -1.230000 | 0.000000 | 14223 |
| angle_hinge | -0.090650 | 0.090650 | -31.026371 | 0.000000 | 14223 |
| progress_shaping | 1.080113 | 1.080661 | -0.517164 | 1.635985 | 14223 |
| total_reward | 0.894869 | 1.071979 | -31.840402 | 1.632952 | 14223 |
