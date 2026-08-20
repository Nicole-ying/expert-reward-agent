# Training Feedback

## Training outcome
score=-105.897031, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.015973 | 0.016913 | 0.999995 | 0.015973 |
| soft_landing_proxy | 0.001721 | 0.001721 | 0.003442 | 0.001721 |
| stability_penalty | -0.014131 | 0.014131 | 1.000000 | -0.014131 |
| total_reward | 0.003563 | 0.009544 | 1.000000 | 0.003563 |
| generated_reward | 0.003563 | 0.009544 | 1.000000 | 0.003563 |
| original_env_reward | -1.521290 | 2.439374 | 1.000000 | -1.521290 |

## Distribution
- score: mean=-105.897031, min=-122.962817, max=-96.298785
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
