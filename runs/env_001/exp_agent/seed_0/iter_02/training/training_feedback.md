# Training Feedback

## Training outcome
score=-114.991403, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.001361 | 0.001361 | 0.136142 | -0.084096 |
| progress_delta_reward | 0.016189 | 0.017098 | 0.999998 | 1.000000 |
| soft_landing_proxy | 0.008516 | 0.008516 | 0.007238 | 0.526044 |
| stability_penalty | -0.006570 | 0.006570 | 1.000000 | -0.405838 |
| total_reward | 0.016773 | 0.020039 | 1.000000 | 1.036110 |
| generated_reward | 0.016773 | 0.020039 | 1.000000 | 1.036110 |
| original_env_reward | -1.622215 | 2.382963 | 1.000000 | -100.205342 |

## Distribution
- score: mean=-114.991403, min=-127.404577, max=-104.924030
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
