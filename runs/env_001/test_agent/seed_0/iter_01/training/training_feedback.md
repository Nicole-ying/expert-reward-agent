# Training Feedback

## Training outcome
score=-111.554261, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.051732 | 0.055383 | 1.000000 | 1.000000 |
| soft_landing_proxy | 0.004579 | 0.004579 | 0.004579 | 0.088519 |
| stability_penalty | -0.330070 | 0.330070 | 1.000000 | -6.380333 |
| total_reward | -0.273758 | 0.282642 | 1.000000 | -5.291815 |
| generated_reward | -0.273758 | 0.282642 | 1.000000 | -5.291815 |
| original_env_reward | -1.513035 | 2.509036 | 1.000000 | -29.247325 |

## Distribution
- score: mean=-111.554261, min=-123.301028, max=-97.900804
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
