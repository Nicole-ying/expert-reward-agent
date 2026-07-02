# Training Feedback

## Training outcome
score=235.963455, len=478.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| bounded_proximity | 0.559019 | 0.559019 | 1.000000 | 0.559019 |
| soft_landing_bonus | 0.156168 | 0.156168 | 0.512087 | 0.156168 |
| stability_penalty | 0.003080 | 0.003080 | 1.000000 | 0.003080 |
| total_reward | 0.712108 | 0.712108 | 1.000000 | 0.712108 |
| generated_reward | 0.712108 | 0.712108 | 1.000000 | 0.712108 |
| original_env_reward | -0.006798 | 1.513236 | 1.000000 | -0.006798 |

## Distribution
- score: mean=235.963455, min=161.961398, max=283.077842
- episode_length: mean=478.900000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
