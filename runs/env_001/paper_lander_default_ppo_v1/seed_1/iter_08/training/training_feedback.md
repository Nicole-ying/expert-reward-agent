# Training Feedback

## Final-policy outcome
score=144.110012, len=305.300000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-551.878418, 285.670028]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_proxy | 643.314443 | 87.1% | 87.1% | 93.9% |
| speed_tracking_reward | -90.962389 | -12.3% | 12.3% | 100.0% |
| fuel_penalty | -2.579000 | -0.3% | 0.3% | 84.5% |
| progress_reward | 1.197511 | 0.2% | 0.2% | 97.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
