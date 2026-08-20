# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.forward_progress | 0.595240 | 0.595240 | 0.582468 | 1.021928 | 1.021928 | 0.000000 | 4.712516 | 1001472 |
| component.height_boundary_hinge | -0.008466 | 0.008466 | 0.273049 | -0.031004 | 0.031004 | -0.079980 | -0.000000 | 1001472 |
| component.lateral_drift_hinge | -0.088895 | 0.088895 | 0.444101 | -0.200168 | 0.200168 | -2.109714 | -0.000000 | 1001472 |
| component.roll_pitch_hinge | -0.084292 | 0.084292 | 0.429047 | -0.196463 | 0.196463 | -3.155825 | -0.000000 | 1001472 |
| component.total_reward | 0.413587 | 0.580912 | 0.959189 | 0.431184 | 0.605628 | -3.422421 | 4.557858 | 1001472 |
| generated_reward | 0.413587 | 0.580912 | 0.959189 | 0.431184 | 0.605628 | -3.422421 | 4.557858 | 1001472 |
| original_env_reward | -0.807120 | 1.117002 | 1.000000 | -0.807120 | 1.117002 | -5.898156 | 4.180755 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| forward_progress | 211.494926 | 211.494926 | 0.000000 | 1522.945979 | 2815 |
| height_boundary_hinge | -3.011701 | 3.011701 | -31.053540 | 0.000000 | 2815 |
| lateral_drift_hinge | -31.605024 | 31.605024 | -167.946073 | 0.000000 | 2815 |
| roll_pitch_hinge | -29.974763 | 29.974763 | -194.486765 | -0.675512 | 2815 |
| total_reward | 146.903437 | 172.911121 | -273.213868 | 1364.698145 | 2815 |
