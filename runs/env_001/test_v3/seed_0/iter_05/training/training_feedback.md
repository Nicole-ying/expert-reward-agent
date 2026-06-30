# Training Feedback

## External evaluation
- score: -190.674320
- episode_length: 84.300000 (mean)
- range: [-260.651416, -3.140113]
- errors: 0

## Component evidence

| component | mean | abs_mean | nonzero_rate | min | max |
|-----------|------|----------|-------------|-----|-----|
| landing_shaping | 0.009739 | 0.009739 | 0.002191 | 0.000000 | 9.571960 |
| progress_reward | 5.118749 | 5.338730 | 0.999999 | -9.893801 | 18.333101 |
| stability_penalty | -0.067521 | 0.067521 | 1.000000 | -0.265954 | -0.000000 |
| total_reward | 5.060968 | 5.282389 | 1.000000 | -9.973926 | 18.097342 |
| generated_reward | 5.060968 | 5.282389 | 1.000000 | -9.973926 | 18.097342 |
| original_env_reward | -2.549996 | 4.094505 | 1.000000 | -100.000000 | 144.022630 |

## Signals
early_failure_or_crash; sparse_proxy:landing_shaping; penalty_dominance:generated_reward
