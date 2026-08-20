# Training Feedback

## Training outcome
score=-54.335404, len=76.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.015284 | 0.016240 | 0.999995 | 1.000000 |
| soft_landing_proxy | 0.002613 | 0.002613 | 0.993934 | 0.170947 |
| stability_penalty | -0.013359 | 0.013359 | 1.000000 | -0.874052 |
| total_reward | 0.004538 | 0.008765 | 1.000000 | 0.296895 |
| generated_reward | 0.004538 | 0.008765 | 1.000000 | 0.296895 |
| original_env_reward | -0.971416 | 2.758795 | 1.000000 | -63.557543 |

## Distribution
- score: mean=-54.335404, min=-91.502316, max=-19.793447
- episode_length: mean=76.200000
- early_terminal (<150 steps + score<-50): 6/10 (60%)
- errors: 0
