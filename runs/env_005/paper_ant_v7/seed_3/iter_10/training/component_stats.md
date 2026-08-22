# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.gated_forward | 0.080544 | 0.080544 | 0.568109 | 0.141775 | 0.141775 | 0.000000 | 0.836689 | 1001472 |
| component.lateral_gate | 0.403578 | 0.403578 | 1.000000 | 0.403578 | 0.403578 | 0.000000 | 0.999998 | 1001472 |
| component.total_reward | -0.830847 | 0.960741 | 0.999992 | -0.830853 | 0.960748 | -5.000000 | 0.824200 | 1001472 |
| component.upright_penalty | 0.911391 | 0.911391 | 0.999984 | 0.911405 | 0.911405 | 0.000000 | 5.000000 | 1001472 |
| generated_reward | -0.830847 | 0.960741 | 0.999992 | -0.830853 | 0.960748 | -5.000000 | 0.824200 | 1001472 |
| original_env_reward | -1.035853 | 1.339110 | 1.000000 | -1.035853 | 1.339110 | -7.378435 | 5.351701 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| gated_forward | 1.758615 | 1.758615 | 0.000000 | 61.716000 | 45867 |
| lateral_gate | 8.811834 | 8.811834 | 0.873485 | 620.138120 | 45867 |
| total_reward | -18.140922 | 20.080644 | -4894.194103 | 18.818001 | 45867 |
| upright_penalty | 19.899537 | 19.899537 | 0.000026 | 4940.650866 | 45867 |
