# Training Feedback

## Training outcome
score=-108.814423, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.006799 | 0.006799 | 0.135972 | -0.042126 |
| landing_bonus | 0.012915 | 0.012915 | 0.006457 | 0.080021 |
| progress_reward | 0.161388 | 0.170750 | 0.999990 | 1.000000 |
| stability_penalty | -0.129303 | 0.129303 | 1.000000 | -0.801193 |
| total_reward | 0.038201 | 0.109547 | 1.000000 | 0.236702 |
| generated_reward | 0.038201 | 0.109547 | 1.000000 | 0.236702 |
| original_env_reward | -1.577753 | 2.299391 | 1.000000 | -9.776124 |

## Distribution
- score: mean=-108.814423, min=-121.458652, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
