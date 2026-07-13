# Training Feedback

## Final-policy outcome
score=-96.500279, len=80.200000, terminated=5/5, truncated=0/5, reward_errors=0
score_range=[-97.216587, -93.963102]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 10.621309 | 51.0% | 51.2% | 100.0% |
| stability_penalty | -10.159233 | -48.8% | 48.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10004
- early_terminal (<150 steps and score<-50): 5/5
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
