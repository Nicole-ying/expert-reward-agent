# Training Feedback

## Training outcome
score=242.406225, len=481.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| dist_gate | 0.526483 | 0.526483 | 1.000000 | 168.653546 |
| landing_proxy | 0.447489 | 0.447489 | 0.561646 | 143.348474 |
| progress_reward | 0.003122 | 0.003423 | 0.999584 | 1.000000 |
| stability_penalty | -0.001045 | 0.001045 | 1.000000 | -0.334648 |
| total_reward | 0.449566 | 0.450037 | 1.000000 | 144.013826 |
| generated_reward | 0.449566 | 0.450037 | 1.000000 | 144.013826 |
| original_env_reward | -0.038528 | 1.560899 | 1.000000 | -12.342087 |

## Distribution
- score: mean=242.406225, min=109.652525, max=282.466855
- episode_length: mean=481.800000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
