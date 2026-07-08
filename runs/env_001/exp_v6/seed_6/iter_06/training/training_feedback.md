# Training Feedback

## Training outcome
score=148.127896, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016634 | 0.017768 | 0.999788 | 1.000000 |
| soft_landing_proxy | 0.206669 | 0.206669 | 0.695755 | 12.424188 |
| stability_penalty | -0.000870 | 0.000870 | 1.000000 | -0.052276 |
| total_reward | 0.222434 | 0.223353 | 1.000000 | 13.371912 |
| generated_reward | 0.222434 | 0.223353 | 1.000000 | 13.371912 |
| original_env_reward | -0.062854 | 1.633194 | 1.000000 | -3.778527 |

## Distribution
- score: mean=148.127896, min=115.214979, max=174.221604
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
