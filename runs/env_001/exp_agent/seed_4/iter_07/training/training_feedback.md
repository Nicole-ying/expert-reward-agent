# Training Feedback

## Training outcome
score=252.324383, len=245.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.001956 | 0.002199 | 0.999713 | 1.000000 |
| soft_landing_proxy | 0.024084 | 0.024084 | 0.990809 | 12.315602 |
| stability_penalty | -0.000589 | 0.000589 | 1.000000 | -0.300940 |
| total_reward | 0.025451 | 0.025526 | 1.000000 | 13.014662 |
| generated_reward | 0.025451 | 0.025526 | 1.000000 | 13.014662 |
| original_env_reward | 0.047978 | 2.392188 | 1.000000 | 24.533717 |

## Distribution
- score: mean=252.324383, min=44.572821, max=293.834767
- episode_length: mean=245.600000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
