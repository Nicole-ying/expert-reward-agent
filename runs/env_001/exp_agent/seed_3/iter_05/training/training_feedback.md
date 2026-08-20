# Training Feedback

## Training outcome
score=-110.742220, len=70.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.016137 | 0.017054 | 0.999999 | 1.000000 |
| soft_landing_proxy | 0.001456 | 0.001456 | 0.006490 | 0.090250 |
| stability_penalty | -0.011195 | 0.011195 | 1.000000 | -0.693739 |
| total_reward | 0.151631 | 0.162919 | 1.000000 | 9.396511 |
| generated_reward | 0.151631 | 0.162919 | 1.000000 | 9.396511 |
| original_env_reward | -1.602028 | 2.425117 | 1.000000 | -99.276744 |

## Distribution
- score: mean=-110.742220, min=-124.592751, max=-97.569959
- episode_length: mean=70.600000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0
