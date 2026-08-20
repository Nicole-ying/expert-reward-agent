# Training Feedback

## Training outcome
score=148.741576, len=745.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.011133 | 0.011896 | 0.999968 | 1.000000 |
| soft_landing_proxy | 0.017456 | 0.017456 | 0.087281 | 1.567901 |
| stability_penalty | -0.011810 | 0.011810 | 1.000000 | -1.060758 |
| total_reward | 0.116981 | 0.127337 | 1.000000 | 10.507143 |
| generated_reward | 0.116981 | 0.127337 | 1.000000 | 10.507143 |
| original_env_reward | -0.691319 | 2.784432 | 1.000000 | -62.093885 |

## Distribution
- score: mean=148.741576, min=25.154797, max=275.040835
- episode_length: mean=745.500000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
