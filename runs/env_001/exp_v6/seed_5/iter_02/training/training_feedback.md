# Training Feedback

## Training outcome
score=-89.182935, len=70.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_proxy | 0.004930 | 0.004930 | 0.011068 | 0.308166 |
| progress_reward | 0.015997 | 0.016930 | 0.999996 | 1.000000 |
| stability_penalty | -0.011363 | 0.011363 | 1.000000 | -0.710304 |
| total_reward | 0.009564 | 0.013537 | 1.000000 | 0.597862 |
| generated_reward | 0.009564 | 0.013537 | 1.000000 | 0.597862 |
| original_env_reward | -1.510221 | 2.485527 | 1.000000 | -94.404477 |

## Distribution
- score: mean=-89.182935, min=-117.722997, max=-52.225025
- episode_length: mean=70.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
