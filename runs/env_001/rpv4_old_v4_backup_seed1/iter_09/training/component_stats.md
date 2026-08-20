# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.goal_proximity | 0.010337 | 0.013165 | 0.999887 | 0.010338 | 0.013166 | -0.143290 | 0.152429 | 1003520 |
| component.soft_landing_envelope | 1.624945 | 1.624945 | 1.000000 | 1.624945 | 1.624945 | 0.000009 | 3.000000 | 1003520 |
| component.total_reward | 1.517679 | 1.625396 | 1.000000 | 1.517679 | 1.625396 | -20.000000 | 3.000472 | 1003520 |
| component.upright_penalty | -0.066911 | 0.066911 | 0.999989 | -0.066912 | 0.066912 | -55.368702 | -0.000000 | 1003520 |
| component.velocity_penalty | -0.052613 | 0.052613 | 0.999866 | -0.052620 | 0.052620 | -3.435310 | -0.000000 | 1003520 |
| generated_reward | 1.517679 | 1.625396 | 1.000000 | 1.517679 | 1.625396 | -20.000000 | 3.000472 | 1003520 |
| original_env_reward | -0.082527 | 2.793515 | 1.000000 | -0.082527 | 2.793515 | -100.000000 | 130.970241 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| goal_proximity | 4.386876 | 4.386876 | 0.511917 | 5.681248 | 2361 |
| soft_landing_envelope | 688.963264 | 688.963264 | 0.325779 | 2631.535759 | 2361 |
| total_reward | 643.420701 | 681.021612 | -311.080035 | 2612.748254 | 2361 |
| upright_penalty | -28.405374 | 28.405374 | -262.078523 | -0.123746 | 2361 |
| velocity_penalty | -22.340962 | 22.340962 | -86.750112 | -5.275339 | 2361 |
