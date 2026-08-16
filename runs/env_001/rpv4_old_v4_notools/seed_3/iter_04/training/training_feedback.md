# Training Feedback

## Final-policy outcome
score=-111.618457, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.407745, -95.638888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_progress | 69.000000 | 86.3% | 86.3% | 2.6% |
| goal_proximity | 7.735110 | 9.7% | 10.0% | 100.0% |
| velocity_penalty | -2.656207 | -3.3% | 3.3% | 99.6% |
| orientation_penalty | -0.276727 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
