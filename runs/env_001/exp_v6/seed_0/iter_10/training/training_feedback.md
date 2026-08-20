# Training Feedback

## Training outcome
score=-228.668948, len=80.700000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_reward | 0.085492 | 0.088697 | 0.999998 | 1.000000 |
| soft_landing_proxy | 0.000001 | 0.000001 | 0.003755 | 0.000016 |
| stability_penalty | -0.002169 | 0.002169 | 1.000000 | -0.025366 |
| total_reward | 0.083325 | 0.086742 | 1.000000 | 0.974650 |
| generated_reward | 0.083325 | 0.086742 | 1.000000 | 0.974650 |
| original_env_reward | -2.295761 | 3.947092 | 1.000000 | -26.853540 |

## Distribution
- score: mean=-228.668948, min=-254.943826, max=-191.307815
- episode_length: mean=80.700000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
