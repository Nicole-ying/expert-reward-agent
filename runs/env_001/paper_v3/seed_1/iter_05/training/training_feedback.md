# Training Feedback

## Final-policy outcome
score=1.594058, len=964.850000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-39.617682, 203.718993]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -108.560965 | -87.1% | 87.1% | 100.0% |
| velocity_penalty | -8.792170 | -7.1% | 7.1% | 100.0% |
| contact_reward | 6.214309 | 5.0% | 5.0% | 0.2% |
| orientation_penalty | -1.035188 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
