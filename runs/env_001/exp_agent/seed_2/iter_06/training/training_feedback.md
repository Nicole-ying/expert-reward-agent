# Training Feedback

## Training outcome
score=-116.949772, len=71.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005432 | 0.005432 | 0.099739 | -0.033587 |
| progress_delta_reward | 0.161717 | 0.170881 | 0.999994 | 1.000000 |
| soft_landing_bonus | 0.010848 | 0.010848 | 0.005424 | 0.067079 |
| stability_penalty | -0.033476 | 0.033476 | 1.000000 | -0.207002 |
| total_reward | 0.133657 | 0.152343 | 1.000000 | 0.826490 |
| generated_reward | 0.133657 | 0.152343 | 1.000000 | 0.826490 |
| original_env_reward | -1.658587 | 2.353407 | 1.000000 | -10.256135 |

## Distribution
- score: mean=-116.949772, min=-143.017581, max=-103.429116
- episode_length: mean=71.800000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
