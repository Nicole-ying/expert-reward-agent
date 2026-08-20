# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.orientation_penalty | -0.003209 | 0.003209 | 0.999910 | -0.003210 | 0.003210 | -2.904351 | -0.000000 | 1003520 |
| component.safe_progress | 0.003097 | 0.003097 | 0.664454 | 0.004661 | 0.004661 | 0.000000 | 0.027923 | 1003520 |
| component.soft_landing | 0.473843 | 0.473843 | 0.536825 | 0.882676 | 0.882676 | 0.000000 | 1.000000 | 1003520 |
| component.total_reward | 0.947573 | 0.951191 | 1.000000 | 0.947573 | 0.951191 | -2.904351 | 2.000052 | 1003520 |
| generated_reward | 0.947573 | 0.951191 | 1.000000 | 0.947573 | 0.951191 | -2.904351 | 2.000052 | 1003520 |
| original_env_reward | 0.001928 | 1.496608 | 1.000000 | 0.001928 | 1.496608 | -100.000000 | 129.307018 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| orientation_penalty | -1.165671 | 1.165671 | -29.239363 | -0.006291 | 2762 |
| safe_progress | 1.123849 | 1.123849 | 0.100698 | 1.703866 | 2762 |
| soft_landing | 171.926074 | 171.926074 | 0.000000 | 825.528858 | 2762 |
| total_reward | 343.810326 | 344.454526 | -28.175741 | 1651.725152 | 2762 |
