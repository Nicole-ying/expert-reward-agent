# Training Feedback

## Training outcome
score=-111.677964, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.007854 | 0.007854 | 0.143210 | -0.048738 |
| progress_delta_reward | 0.161144 | 0.170426 | 0.999995 | 1.000000 |
| soft_landing_bonus | 0.010838 | 0.010838 | 0.005419 | 0.067255 |
| stability_penalty | -0.555900 | 0.555900 | 1.000000 | -3.449699 |
| total_reward | -0.391771 | 0.411592 | 1.000000 | -2.431181 |
| generated_reward | -0.391771 | 0.411592 | 1.000000 | -2.431181 |
| original_env_reward | -1.624858 | 2.347336 | 1.000000 | -10.083237 |

## Distribution
- score: mean=-111.677964, min=-124.079149, max=-95.059093
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
