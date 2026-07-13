# Training Feedback

## Final-policy outcome
score=-30.271718, len=946.650000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-83.684064, 41.031883]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 13.700792 | 56.3% | 56.3% | 0.8% |
| velocity_penalty | -7.480576 | -30.7% | 30.7% | 100.0% |
| goal_proximity | 1.894493 | 7.8% | 8.8% | 100.0% |
| orientation_penalty | -1.021556 | -4.2% | 4.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
