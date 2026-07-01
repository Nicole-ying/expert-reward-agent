# Training Feedback

## Training outcome
score=242.086713, len=441.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angular_vel_penalty | -0.000592 | 0.000592 | 0.962926 | -0.000592 |
| energy_penalty | -0.007546 | 0.007546 | 0.754637 | -0.007546 |
| potential_shaping | 0.021351 | 0.030130 | 0.999998 | 0.021351 |
| soft_landing_proxy | 0.829484 | 0.829484 | 0.982930 | 0.829484 |
| total_reward | 0.842697 | 0.842805 | 1.000000 | 0.842697 |
| generated_reward | 0.842697 | 0.842805 | 1.000000 | 0.842697 |
| original_env_reward | -0.117375 | 1.916126 | 1.000000 | -0.117375 |

## Distribution
- score: mean=242.086713, min=216.675382, max=277.852200
- episode_length: mean=441.300000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
