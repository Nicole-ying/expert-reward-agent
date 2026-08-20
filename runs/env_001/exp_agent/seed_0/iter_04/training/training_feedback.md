# Training Feedback

## Training outcome
score=261.780150, len=284.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.003556 | 0.003911 | 0.998337 | 1.000000 |
| soft_landing_proxy | 0.484643 | 0.484643 | 0.713254 | 136.297445 |
| stability_penalty | -0.000646 | 0.000646 | 0.999985 | -0.181692 |
| total_reward | 0.501775 | 0.502800 | 1.000000 | 141.115752 |
| generated_reward | 0.501775 | 0.502800 | 1.000000 | 141.115752 |
| original_env_reward | -0.143712 | 3.184503 | 1.000000 | -40.416600 |

## Distribution
- score: mean=261.780150, min=232.724231, max=291.172646
- episode_length: mean=284.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
