# Training Feedback

## Training outcome
score=-110.683396, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.008435 | 0.008435 | 0.168709 | -0.052693 |
| progress_delta_reward | 0.160088 | 0.169472 | 0.999990 | 1.000000 |
| soft_landing_bonus | 0.011517 | 0.011517 | 0.005759 | 0.071945 |
| stability_penalty | -0.242176 | 0.242176 | 1.000000 | -1.512771 |
| total_reward | -0.079006 | 0.105815 | 1.000000 | -0.493519 |
| generated_reward | -0.079006 | 0.105815 | 1.000000 | -0.493519 |
| original_env_reward | -1.545106 | 2.316389 | 1.000000 | -9.651624 |

## Distribution
- score: mean=-110.683396, min=-130.746612, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
