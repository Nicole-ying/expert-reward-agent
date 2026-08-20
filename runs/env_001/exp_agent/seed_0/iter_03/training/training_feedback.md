# Training Feedback

## Training outcome
score=-1771.110673, len=213.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angular_vel_penalty | -0.001080 | 0.001080 | 0.999997 | -0.001080 |
| energy_penalty | -0.020271 | 0.020271 | 0.999996 | -0.020271 |
| potential_shaping | 0.130193 | 0.130394 | 0.999998 | 0.130193 |
| soft_landing_proxy | 0.000705 | 0.000705 | 0.001510 | 0.000705 |
| total_reward | 0.110252 | 0.113534 | 1.000000 | 0.110252 |
| generated_reward | 0.110252 | 0.113534 | 1.000000 | 0.110252 |
| original_env_reward | -4.382563 | 4.911105 | 1.000000 | -4.382563 |

## Distribution
- score: mean=-1771.110673, min=-5945.060097, max=-574.418700
- episode_length: mean=213.500000
- early_terminal (<150 steps + score<-50): 6/10 (60%)
- errors: 0
