# Training Feedback

## Training outcome
score=142.404922, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.002176 | 0.002465 | 0.999509 | 1.000000 |
| soft_landing_proxy | 0.329776 | 0.329776 | 0.981201 | 151.524846 |
| stability_penalty | -0.004463 | 0.004463 | 1.000000 | -2.050668 |
| total_reward | 0.347076 | 0.347398 | 1.000000 | 159.474178 |
| generated_reward | 0.347076 | 0.347398 | 1.000000 | 159.474178 |
| original_env_reward | 0.037073 | 1.280289 | 1.000000 | 17.034340 |

## Distribution
- score: mean=142.404922, min=116.852775, max=177.592067
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
