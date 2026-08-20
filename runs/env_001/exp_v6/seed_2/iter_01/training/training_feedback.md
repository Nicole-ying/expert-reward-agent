# Training Feedback

## Training outcome
score=-108.614223, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_penalty | -0.048590 | 0.048590 | 1.000000 | -3.023055 |
| progress_delta_reward | 0.016073 | 0.017006 | 0.999993 | 1.000000 |
| soft_landing_bonus | 0.000532 | 0.000532 | 0.000532 | 0.033107 |
| stability_penalty | -0.081839 | 0.081839 | 1.000000 | -5.091688 |
| total_reward | -0.113824 | 0.114840 | 1.000000 | -7.081637 |
| generated_reward | -0.113824 | 0.114840 | 1.000000 | -7.081637 |
| original_env_reward | -1.525328 | 2.369727 | 1.000000 | -94.899531 |

## Distribution
- score: mean=-108.614223, min=-120.796346, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
