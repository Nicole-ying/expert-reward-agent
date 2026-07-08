# Training Feedback

## Training outcome
score=-107.157448, len=69.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016190 | 0.017119 | 0.999994 | 1.000000 |
| soft_landing_proxy | 0.001381 | 0.001381 | 1.000000 | 0.085321 |
| stability_penalty | -0.001489 | 0.001489 | 1.000000 | -0.091998 |
| total_reward | 0.016082 | 0.016969 | 1.000000 | 0.993322 |
| generated_reward | 0.016082 | 0.016969 | 1.000000 | 0.993322 |
| original_env_reward | -1.549250 | 2.439284 | 1.000000 | -95.692719 |

## Distribution
- score: mean=-107.157448, min=-122.475523, max=-91.850620
- episode_length: mean=69.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
