# Training Feedback

## Training outcome
score=146.134454, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.044699 | 0.047495 | 0.998672 | 1.000000 |
| soft_landing_proxy | 0.206022 | 0.206022 | 0.503618 | 4.609094 |
| stability_penalty | -0.000622 | 0.000622 | 1.000000 | -0.013921 |
| total_reward | 0.250098 | 0.252532 | 1.000000 | 5.595173 |
| generated_reward | 0.250098 | 0.252532 | 1.000000 | 5.595173 |
| original_env_reward | -0.217021 | 1.674786 | 1.000000 | -4.855179 |

## Distribution
- score: mean=146.134454, min=122.832738, max=176.555201
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
