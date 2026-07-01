# Training Feedback

## Training outcome
score=179.442777, len=812.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angular_vel_penalty | -0.000561 | 0.000561 | 0.945943 | -0.000561 |
| energy_penalty | -0.007266 | 0.007266 | 0.726572 | -0.007266 |
| potential_shaping | 0.045239 | 0.060897 | 0.999998 | 0.045239 |
| soft_landing_proxy | 0.336278 | 0.336278 | 0.976892 | 0.336278 |
| total_reward | 0.373691 | 0.373996 | 1.000000 | 0.373691 |
| generated_reward | 0.373691 | 0.373996 | 1.000000 | 0.373691 |
| original_env_reward | -0.012887 | 1.355248 | 1.000000 | -0.012887 |

## Distribution
- score: mean=179.442777, min=123.414567, max=267.990132
- episode_length: mean=812.600000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
