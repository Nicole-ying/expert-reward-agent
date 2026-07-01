# Training Feedback

## Training outcome
score=-118.437388, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.004933 | 0.004933 | 0.098654 | -0.030456 |
| progress_delta_reward | 0.161961 | 0.171098 | 0.999998 | 1.000000 |
| soft_landing_bonus | 0.013810 | 0.013810 | 0.015705 | 0.085266 |
| stability_penalty | -0.006025 | 0.006025 | 1.000000 | -0.037200 |
| total_reward | 0.164814 | 0.176646 | 1.000000 | 1.017610 |
| generated_reward | 0.164814 | 0.176646 | 1.000000 | 1.017610 |
| original_env_reward | -1.722980 | 2.425202 | 1.000000 | -10.638209 |

## Distribution
- score: mean=-118.437388, min=-140.549214, max=-105.349835
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
