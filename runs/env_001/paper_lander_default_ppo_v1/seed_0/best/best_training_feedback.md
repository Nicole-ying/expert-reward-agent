# Training Feedback

## Final-policy outcome
score=240.078868, len=231.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-45.156158, 302.814134]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_shaping_reward | 59.991357 | 79.1% | 98.0% | 100.0% |
| progress_reward | 1.377250 | 1.8% | 1.9% | 94.8% |
| stability_penalty | -0.123381 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
