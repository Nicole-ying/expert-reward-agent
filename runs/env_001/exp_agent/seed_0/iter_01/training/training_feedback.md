# Training Feedback

## Training outcome
score=-115.524472, len=74.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.001737 | 0.001737 | 0.173727 | -0.053713 |
| progress_delta_reward | 0.032344 | 0.034174 | 0.999996 | 1.000000 |
| soft_landing_proxy | 0.005204 | 0.005204 | 0.005204 | 0.160886 |
| stability_penalty | -0.108909 | 0.108909 | 1.000000 | -3.367223 |
| total_reward | -0.073099 | 0.083249 | 1.000000 | -2.260049 |
| generated_reward | -0.073099 | 0.083249 | 1.000000 | -2.260049 |
| original_env_reward | -1.581795 | 2.341473 | 1.000000 | -48.905611 |

## Distribution
- score: mean=-115.524472, min=-130.746612, max=-105.791517
- episode_length: mean=74.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
