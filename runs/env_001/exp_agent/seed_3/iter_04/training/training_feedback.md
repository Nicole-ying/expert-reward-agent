# Training Feedback

## Training outcome
score=-111.552558, len=70.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.016147 | 0.017074 | 0.999993 | 1.000000 |
| soft_landing_proxy | 0.000846 | 0.000846 | 0.006256 | 0.052368 |
| stability_penalty | -0.011309 | 0.011309 | 1.000000 | -0.700407 |
| total_reward | 0.151006 | 0.162492 | 1.000000 | 9.351962 |
| generated_reward | 0.151006 | 0.162492 | 1.000000 | 9.351962 |
| original_env_reward | -1.610148 | 2.431917 | 1.000000 | -99.718274 |

## Distribution
- score: mean=-111.552558, min=-124.463377, max=-98.869122
- episode_length: mean=70.600000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
