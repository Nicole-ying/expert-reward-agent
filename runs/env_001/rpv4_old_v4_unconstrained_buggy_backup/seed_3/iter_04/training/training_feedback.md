# Training Feedback

## Final-policy outcome
score=-109.533802, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-137.319938, -67.062972]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 142.400000 | 92.4% | 92.4% | 1.8% |
| velocity_penalty | -7.935640 | -5.2% | 5.2% | 99.6% |
| goal_proximity | -3.328360 | -2.2% | 2.2% | 100.0% |
| orientation_penalty | -0.387919 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
