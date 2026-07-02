# Training Feedback

## Training outcome
score=-108.944442, len=73.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.016063 | 0.017000 | 0.999990 | 1.000000 |
| soft_landing_bonus | 0.001067 | 0.001067 | 0.002134 | 0.066441 |
| stability_penalty | 0.147474 | 0.147474 | 1.000000 | 9.181011 |
| total_reward | -0.130344 | 0.132426 | 1.000000 | -8.114570 |
| generated_reward | -0.130344 | 0.132426 | 1.000000 | -8.114570 |
| original_env_reward | -1.519302 | 2.389056 | 1.000000 | -94.584410 |

## Distribution
- score: mean=-108.944442, min=-121.621397, max=-95.428352
- episode_length: mean=73.600000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
