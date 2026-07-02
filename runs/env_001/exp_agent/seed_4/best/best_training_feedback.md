# Training Feedback

## Training outcome
score=242.224287, len=624.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002184 | 0.002364 | 0.999483 | 1.000000 |
| soft_landing_proxy | 0.024061 | 0.024061 | 0.993512 | 11.016659 |
| stability_penalty | -0.001867 | 0.001867 | 1.000000 | -0.854788 |
| total_reward | 0.024379 | 0.024603 | 1.000000 | 11.161870 |
| generated_reward | 0.024379 | 0.024603 | 1.000000 | 11.161870 |
| original_env_reward | 0.055818 | 1.986136 | 1.000000 | 25.556424 |

## Distribution
- score: mean=242.224287, min=137.361742, max=293.484305
- episode_length: mean=624.500000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
