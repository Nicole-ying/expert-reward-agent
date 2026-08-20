# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.descent_early_penalty | -0.000900 | 0.000900 | 0.229222 | -0.003926 | 0.003926 | -0.244173 | 0.000000 | 1001472 |
| component.forward_progress | 0.694092 | 0.694092 | 0.566670 | 1.224862 | 1.224862 | 0.000000 | 5.429902 | 1001472 |
| component.lateral_drift_hinge | -0.086410 | 0.086410 | 0.434283 | -0.198972 | 0.198972 | -2.083030 | -0.000000 | 1001472 |
| component.roll_pitch_hinge | -0.084793 | 0.084793 | 0.439046 | -0.193130 | 0.193130 | -3.155825 | -0.000000 | 1001472 |
| component.total_reward | 0.521989 | 0.642129 | 0.850063 | 0.614060 | 0.755390 | -3.300014 | 4.766715 | 1001472 |
| generated_reward | 0.521989 | 0.642129 | 0.850063 | 0.614060 | 0.755390 | -3.300014 | 4.766715 | 1001472 |
| original_env_reward | -0.781614 | 1.213657 | 1.000000 | -0.781614 | 1.213657 | -5.947684 | 4.647730 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| descent_early_penalty | -0.400392 | 0.400392 | -6.841005 | 0.000000 | 2250 |
| forward_progress | 308.885206 | 308.885206 | 0.000000 | 1753.032307 | 2250 |
| lateral_drift_hinge | -38.455737 | 38.455737 | -161.425059 | 0.000000 | 2250 |
| roll_pitch_hinge | -37.730457 | 37.730457 | -185.071505 | -0.471567 | 2250 |
| total_reward | 232.298620 | 255.717438 | -272.824833 | 1519.658464 | 2250 |
