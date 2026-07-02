# Training Feedback

## Training outcome
score=-110.961396, len=69.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016150 | 0.017077 | 0.999991 | 1.000000 |
| soft_landing_proxy | 0.000404 | 0.000404 | 0.004040 | 0.025015 |
| stability_penalty | -0.014589 | 0.014589 | 1.000000 | -0.903374 |
| total_reward | 0.001964 | 0.008432 | 1.000000 | 0.121640 |
| generated_reward | 0.001964 | 0.008432 | 1.000000 | 0.121640 |
| original_env_reward | -1.534872 | 2.395598 | 1.000000 | -95.041096 |

## Distribution
- score: mean=-110.961396, min=-123.943225, max=-97.253096
- episode_length: mean=69.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
