# Training Feedback

## Training outcome
score=-111.449688, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016099 | 0.017029 | 0.999993 | 1.000000 |
| soft_landing_proxy | 0.003124 | 0.003124 | 0.004934 | 0.194059 |
| stability_penalty | -0.057574 | 0.057574 | 1.000000 | -3.576277 |
| total_reward | 0.057956 | 0.069595 | 1.000000 | 3.599980 |
| generated_reward | 0.057956 | 0.069595 | 1.000000 | 3.599980 |
| original_env_reward | -1.573099 | 2.422405 | 1.000000 | -97.714285 |

## Distribution
- score: mean=-111.449688, min=-123.520882, max=-98.333719
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
