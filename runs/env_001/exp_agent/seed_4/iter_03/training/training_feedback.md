# Training Feedback

## Training outcome
score=165.264900, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002542 | 0.002752 | 0.999666 | 1.000000 |
| soft_landing_proxy | 0.023730 | 0.023730 | 0.991623 | 9.335099 |
| stability_penalty | -0.002815 | 0.002815 | 1.000000 | -1.107501 |
| total_reward | 0.023457 | 0.023986 | 1.000000 | 9.227598 |
| generated_reward | 0.023457 | 0.023986 | 1.000000 | 9.227598 |
| original_env_reward | 0.023377 | 1.835628 | 1.000000 | 9.195853 |

## Distribution
- score: mean=165.264900, min=134.698208, max=195.139252
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
