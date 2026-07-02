# Training Feedback

## Training outcome
score=84.797051, len=823.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| angle_reduction | -0.000084 | 0.000247 | 0.999782 | -0.020247 |
| angvel_reduction | -0.000102 | 0.000402 | 0.999368 | -0.024506 |
| progress_delta | 0.004144 | 0.004387 | 0.999548 | 1.000000 |
| speed_reduction | 0.000029 | 0.000826 | 1.000000 | 0.007020 |
| total_reward | 0.003988 | 0.004483 | 0.999998 | 0.962267 |
| generated_reward | 0.003988 | 0.004483 | 0.999998 | 0.962267 |
| original_env_reward | -0.129463 | 1.699018 | 1.000000 | -31.238671 |

## Distribution
- score: mean=84.797051, min=48.121934, max=126.608144
- episode_length: mean=823.300000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
