# Training Feedback

## Final-policy outcome
score=37.653049, len=895.900000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-141.735094, 138.633443]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 1960.931640 | 94.0% | 94.0% | 100.0% |
| goal_proximity | -109.445230 | -5.2% | 5.2% | 100.0% |
| velocity_penalty | -13.647435 | -0.7% | 0.7% | 100.0% |
| orientation_penalty | -3.071650 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
