# Training Feedback

## Training outcome
score=206.102192, len=498.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.061315 | 0.061315 | 0.613154 | -1.628091 |
| progress_delta_reward | 0.037661 | 0.040800 | 0.996732 | 1.000000 |
| soft_landing_bonus | 0.973998 | 0.973998 | 0.486999 | 25.862305 |
| stability_penalty | -0.116910 | 0.116910 | 1.000000 | -3.104271 |
| total_reward | 0.833433 | 1.071198 | 1.000000 | 22.129943 |
| generated_reward | 0.833433 | 1.071198 | 1.000000 | 22.129943 |
| original_env_reward | -0.045767 | 1.618534 | 1.000000 | -1.215226 |

## Distribution
- score: mean=206.102192, min=112.398817, max=251.917877
- episode_length: mean=498.900000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0
