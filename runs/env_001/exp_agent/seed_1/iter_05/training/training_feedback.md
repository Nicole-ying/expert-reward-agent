# Training Feedback

## Training outcome
score=255.824960, len=440.400000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| bounded_proximity | 0.569403 | 0.569403 | 1.000000 | 0.569403 |
| soft_landing_bonus | 0.084059 | 0.084059 | 0.372803 | 0.084059 |
| stability_penalty | 0.003094 | 0.003094 | 1.000000 | 0.003094 |
| total_reward | 0.650369 | 0.650369 | 1.000000 | 0.650369 |
| generated_reward | 0.650369 | 0.650369 | 1.000000 | 0.650369 |
| original_env_reward | -0.035210 | 2.504261 | 1.000000 | -0.035210 |

## Distribution
- score: mean=255.824960, min=186.015761, max=299.557386
- episode_length: mean=440.400000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
