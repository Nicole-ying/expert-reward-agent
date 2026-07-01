# Training Feedback

## Training outcome
score=-110.337746, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005890 | 0.005890 | 0.117793 | -0.036612 |
| landing_bonus | 0.017613 | 0.017613 | 0.006125 | 0.109489 |
| progress_reward | 0.160869 | 0.170178 | 0.999995 | 1.000000 |
| stability_penalty | -0.014270 | 0.014270 | 1.000000 | -0.088703 |
| total_reward | 0.158323 | 0.173897 | 1.000000 | 0.984174 |
| generated_reward | 0.158323 | 0.173897 | 1.000000 | 0.984174 |
| original_env_reward | -1.613310 | 2.329773 | 1.000000 | -10.028747 |

## Distribution
- score: mean=-110.337746, min=-124.079149, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
