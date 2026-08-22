# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_progress | 0.037006 | 0.045677 | 0.999730 | 0.037016 | 0.045689 | -0.478182 | 0.556681 | 1003520 |
| component.contact_completion | 1.809005 | 1.809005 | 0.500147 | 3.616943 | 3.616943 | 0.000000 | 4.999999 | 1003520 |
| component.orientation_penalty | -0.025681 | 0.025681 | 0.999979 | -0.025681 | 0.025681 | -12.840859 | -0.000000 | 1003520 |
| component.soft_landing_speed_penalty | -0.092136 | 0.092136 | 0.999773 | -0.092156 | 0.092156 | -3.492244 | -0.000000 | 1003520 |
| component.total_reward | 1.728195 | 1.874900 | 1.000000 | 1.728195 | 1.874900 | -13.063365 | 4.999970 | 1003520 |
| generated_reward | 1.728195 | 1.874900 | 1.000000 | 1.728195 | 1.874900 | -13.063365 | 4.999970 | 1003520 |
| original_env_reward | -0.159042 | 1.611397 | 1.000000 | -0.159042 | 1.611397 | -100.000000 | 132.422534 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_progress | 12.936556 | 13.467007 | -21.676346 | 21.284436 | 2865 |
| contact_completion | 631.342937 | 631.342937 | 0.000000 | 4213.826750 | 2865 |
| orientation_penalty | -8.973836 | 8.973836 | -175.677735 | -0.053972 | 2865 |
| soft_landing_speed_penalty | -32.199346 | 32.199346 | -177.837976 | -0.002320 | 2865 |
| total_reward | 603.106311 | 625.957459 | -194.438525 | 4173.913689 | 2865 |
