# Training Feedback

## Training outcome
score=157.713465, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.003265 | 0.003636 | 0.999684 | 1.000000 |
| soft_landing_proxy | 0.022689 | 0.022689 | 0.989783 | 6.949512 |
| stability_penalty | -0.003576 | 0.003576 | 1.000000 | -1.095214 |
| total_reward | 0.022378 | 0.023067 | 1.000000 | 6.854298 |
| generated_reward | 0.022378 | 0.023067 | 1.000000 | 6.854298 |
| original_env_reward | -0.075576 | 2.006077 | 1.000000 | -23.148258 |

## Distribution
- score: mean=157.713465, min=123.040405, max=190.444691
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
