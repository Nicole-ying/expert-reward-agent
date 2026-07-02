# Training Feedback

## Training outcome
score=129.761934, len=829.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002685 | 0.002854 | 0.999578 | 1.000000 |
| soft_landing_proxy | 0.062194 | 0.062194 | 0.621939 | 23.164424 |
| stability_penalty | -0.000553 | 0.000553 | 1.000000 | -0.206047 |
| total_reward | 0.064326 | 0.064594 | 1.000000 | 23.958377 |
| generated_reward | 0.064326 | 0.064594 | 1.000000 | 23.958377 |
| original_env_reward | 0.046482 | 1.165589 | 1.000000 | 17.312532 |

## Distribution
- score: mean=129.761934, min=-5.243113, max=186.366770
- episode_length: mean=829.800000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
