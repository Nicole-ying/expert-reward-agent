# Training Feedback

## Training outcome
score=-82.158480, len=75.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.049863 | 0.053346 | 0.999993 | 1.000000 |
| soft_landing_proxy | 0.004733 | 0.004733 | 0.008360 | 0.094917 |
| stability_penalty | -0.033169 | 0.033169 | 1.000000 | -0.665196 |
| total_reward | 0.021427 | 0.039071 | 1.000000 | 0.429721 |
| generated_reward | 0.021427 | 0.039071 | 1.000000 | 0.429721 |
| original_env_reward | -1.417608 | 2.757829 | 1.000000 | -28.429848 |

## Distribution
- score: mean=-82.158480, min=-106.599747, max=-69.768991
- episode_length: mean=75.200000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
