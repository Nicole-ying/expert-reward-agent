# Training Feedback

## Training outcome
score=-98.037542, len=72.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_penalty | -0.004847 | 0.004847 | 1.000000 | -0.305614 |
| progress_delta_reward | 0.015859 | 0.016807 | 0.999991 | 1.000000 |
| soft_landing_bonus | 0.008602 | 0.008602 | 0.006519 | 0.542391 |
| stability_penalty | -0.008694 | 0.008694 | 1.000000 | -0.548184 |
| total_reward | 0.010921 | 0.018596 | 1.000000 | 0.688593 |
| generated_reward | 0.010921 | 0.018596 | 1.000000 | 0.688593 |
| original_env_reward | -1.477784 | 2.441471 | 1.000000 | -93.181141 |

## Distribution
- score: mean=-98.037542, min=-118.035399, max=-62.408080
- episode_length: mean=72.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
