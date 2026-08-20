# Training Feedback

## Final-policy outcome
score=-116.978160, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-162.432508, 8.953176]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -77.298786 | -80.0% | 80.0% | 100.0% |
| velocity_penalty | -17.256391 | -17.9% | 17.9% | 100.0% |
| contact_reward | 1.590970 | 1.6% | 1.6% | 1.0% |
| orientation_penalty | -0.418499 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
