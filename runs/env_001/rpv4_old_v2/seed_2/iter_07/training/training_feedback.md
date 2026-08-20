# Training Feedback

## Final-policy outcome
score=156.745708, len=577.950000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-53.730864, 251.506443]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact_success | 110.223827 | 87.8% | 87.8% | 17.8% |
| goal_distance_progress | 5.831261 | 4.6% | 5.8% | 99.1% |
| soft_landing | -5.571498 | -4.4% | 4.4% | 99.2% |
| orientation_penalty | -2.558454 | -2.0% | 2.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
