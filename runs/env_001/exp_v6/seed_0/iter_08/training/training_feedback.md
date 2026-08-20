# Training Feedback

## Training outcome
score=152.376089, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.003713 | 0.003941 | 0.999092 | 1.000000 |
| soft_landing_proxy | 0.025532 | 0.025532 | 0.624890 | 6.876839 |
| stability_penalty | -0.000481 | 0.000481 | 1.000000 | -0.129437 |
| total_reward | 0.028764 | 0.029009 | 1.000000 | 7.747402 |
| generated_reward | 0.028764 | 0.029009 | 1.000000 | 7.747402 |
| original_env_reward | -0.119113 | 1.551319 | 1.000000 | -32.082373 |

## Distribution
- score: mean=152.376089, min=128.345126, max=190.342606
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
