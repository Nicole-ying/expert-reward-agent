# Training Feedback

## Training outcome
score=149.707807, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.003376 | 0.003625 | 0.999529 | 1.000000 |
| soft_landing_proxy | 0.017953 | 0.017953 | 0.667874 | 5.317245 |
| stability_penalty | -0.000432 | 0.000432 | 1.000000 | -0.127827 |
| total_reward | 0.020898 | 0.021111 | 1.000000 | 6.189418 |
| generated_reward | 0.020898 | 0.021111 | 1.000000 | 6.189418 |
| original_env_reward | -0.074682 | 1.492816 | 1.000000 | -22.119176 |

## Distribution
- score: mean=149.707807, min=125.984617, max=182.024049
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
