# Training Feedback

## Final-policy outcome
score=59.609695, len=623.400000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-196.233676, 220.965173]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 391.134345 | 90.5% | 90.5% | 16.8% |
| landing_progress | 0.800112 | 0.2% | 9.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
