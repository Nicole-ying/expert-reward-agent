# Training Feedback

## Training outcome
score=124.965291, len=919.400000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.003727 | 0.003973 | 0.999464 | 1.000000 |
| soft_landing_proxy | 0.046488 | 0.046488 | 0.635354 | 12.473117 |
| stability_penalty | -0.000463 | 0.000463 | 1.000000 | -0.124259 |
| total_reward | 0.049752 | 0.049980 | 1.000000 | 13.348858 |
| generated_reward | 0.049752 | 0.049980 | 1.000000 | 13.348858 |
| original_env_reward | -0.093255 | 1.465653 | 1.000000 | -25.020811 |

## Distribution
- score: mean=124.965291, min=-39.682863, max=179.800901
- episode_length: mean=919.400000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
