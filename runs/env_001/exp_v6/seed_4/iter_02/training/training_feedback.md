# Training Feedback

## Training outcome
score=-111.116530, len=71.400000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.016060 | 0.016981 | 0.999993 | 0.016060 |
| soft_landing_proxy | 0.000751 | 0.000751 | 0.003754 | 0.000751 |
| stability_penalty | -0.005795 | 0.005795 | 1.000000 | -0.005795 |
| total_reward | 0.011015 | 0.013335 | 1.000000 | 0.011015 |
| generated_reward | 0.011015 | 0.013335 | 1.000000 | 0.011015 |
| original_env_reward | -1.524781 | 2.427536 | 1.000000 | -1.524781 |

## Distribution
- score: mean=-111.116530, min=-122.935118, max=-95.059093
- episode_length: mean=71.400000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
