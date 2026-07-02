# Training Feedback

## Training outcome
score=-107.097651, len=74.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016108 | 0.017038 | 0.999992 | 1.000000 |
| soft_landing_proxy | 0.004557 | 0.004557 | 0.004557 | 0.282897 |
| stability_penalty | -0.141834 | 0.141834 | 1.000000 | -8.805068 |
| total_reward | -0.056736 | 0.066443 | 1.000000 | -3.522171 |
| generated_reward | -0.056736 | 0.066443 | 1.000000 | -3.522171 |
| original_env_reward | -1.538305 | 2.393403 | 1.000000 | -95.498350 |

## Distribution
- score: mean=-107.097651, min=-114.859921, max=-87.996822
- episode_length: mean=74.200000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
