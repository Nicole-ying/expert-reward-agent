# Training Feedback

## Training outcome
score=-107.334042, len=71.500000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016088 | 0.017018 | 0.999995 | 1.000000 |
| soft_landing_proxy | 0.001366 | 0.001366 | 0.004554 | 0.084921 |
| stability_penalty | -0.013807 | 0.013807 | 1.000000 | -0.858228 |
| total_reward | 0.003647 | 0.009005 | 1.000000 | 0.226693 |
| generated_reward | 0.003647 | 0.009005 | 1.000000 | 0.226693 |
| original_env_reward | -1.535418 | 2.442821 | 1.000000 | -95.440413 |

## Distribution
- score: mean=-107.334042, min=-126.390418, max=-88.117855
- episode_length: mean=71.500000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
