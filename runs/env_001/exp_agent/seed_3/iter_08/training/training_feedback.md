# Training Feedback

## Training outcome
score=94.798593, len=795.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_reward | -0.551514 | 0.551514 | 1.000000 | -0.551514 |
| proximity_bonus | 0.843004 | 0.843004 | 1.000000 | 0.843004 |
| stability_penalty | -0.086605 | 0.086605 | 1.000000 | -0.086605 |
| total_reward | 0.204885 | 0.943687 | 1.000000 | 0.204885 |
| generated_reward | 0.204885 | 0.943687 | 1.000000 | 0.204885 |
| original_env_reward | -0.437237 | 1.944645 | 1.000000 | -0.437237 |

## Distribution
- score: mean=94.798593, min=-13.215230, max=255.369111
- episode_length: mean=795.900000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
