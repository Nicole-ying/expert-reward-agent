# Training Feedback

## Training outcome
score=-111.951924, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.052337 | 0.055832 | 0.999993 | 1.000000 |
| soft_landing_proxy | 0.002620 | 0.002620 | 0.004553 | 0.050061 |
| stability_penalty | -0.033471 | 0.033471 | 1.000000 | -0.639536 |
| total_reward | 0.021485 | 0.038437 | 1.000000 | 0.410525 |
| generated_reward | 0.021485 | 0.038437 | 1.000000 | 0.410525 |
| original_env_reward | -1.552251 | 2.517299 | 1.000000 | -29.659021 |

## Distribution
- score: mean=-111.951924, min=-121.816517, max=-96.864415
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
