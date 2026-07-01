# Training Feedback

## Training outcome
score=-115.024416, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005695 | 0.005695 | 0.113908 | -0.035362 |
| landing_bonus | 0.042785 | 0.042785 | 0.007405 | 0.265647 |
| progress_reward | 0.161058 | 0.170415 | 0.999997 | 1.000000 |
| stability_penalty | -0.014429 | 0.014429 | 1.000000 | -0.089590 |
| total_reward | 0.183718 | 0.199111 | 1.000000 | 1.140695 |
| generated_reward | 0.183718 | 0.199111 | 1.000000 | 1.140695 |
| original_env_reward | -1.617069 | 2.341746 | 1.000000 | -10.040299 |

## Distribution
- score: mean=-115.024416, min=-130.746612, max=-98.057243
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
