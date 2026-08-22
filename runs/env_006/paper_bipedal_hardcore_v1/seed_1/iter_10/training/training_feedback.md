# Training Feedback

## Final-policy outcome
score=-58.436709, len=197.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-78.054646, -37.539403]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 73.386270 | 93.7% | 93.8% | 100.0% |
| action_penalty | -4.079145 | -5.2% | 5.2% | 100.0% |
| stability_angle_penalty | -0.792735 | -1.0% | 1.0% | 15.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 7/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
