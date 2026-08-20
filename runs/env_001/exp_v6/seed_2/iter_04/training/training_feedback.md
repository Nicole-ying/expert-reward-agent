# Training Feedback

## Training outcome
score=-60.670212, len=85.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| approach_bonus | 0.005273 | 0.005273 | 0.863134 | 0.394949 |
| distance_penalty | -0.004485 | 0.004485 | 1.000000 | -0.335935 |
| progress_delta_reward | 0.013351 | 0.014267 | 0.999994 | 1.000000 |
| stability_penalty | -0.007391 | 0.007391 | 1.000000 | -0.553578 |
| total_reward | 0.006748 | 0.012935 | 1.000000 | 0.505435 |
| generated_reward | 0.006748 | 0.012935 | 1.000000 | 0.505435 |
| original_env_reward | -0.584812 | 2.886165 | 1.000000 | -43.804237 |

## Distribution
- score: mean=-60.670212, min=-108.662357, max=-3.370501
- episode_length: mean=85.300000
- early_terminal (<150 steps + score<-50): 6/10 (60%)
- errors: 0
