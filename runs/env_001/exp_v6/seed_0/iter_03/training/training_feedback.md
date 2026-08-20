# Training Feedback

## Training outcome
score=142.675511, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.045813 | 0.048503 | 0.998634 | 1.000000 |
| soft_landing_proxy | 0.220512 | 0.220512 | 0.441024 | 4.813341 |
| stability_penalty | -0.000613 | 0.000613 | 1.000000 | -0.013382 |
| total_reward | 0.265711 | 0.268230 | 1.000000 | 5.799959 |
| generated_reward | 0.265711 | 0.268230 | 1.000000 | 5.799959 |
| original_env_reward | -0.193894 | 1.740157 | 1.000000 | -4.232325 |

## Distribution
- score: mean=142.675511, min=120.986523, max=175.143642
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
