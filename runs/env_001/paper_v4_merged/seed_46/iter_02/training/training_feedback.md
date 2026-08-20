# Training Feedback

## Final-policy outcome
score=-122.058046, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.209082, -103.429116]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity_progress | 1.127305 | 92.2% | 95.2% | 100.0% |
| terminal_success_bonus | 0.050000 | 4.1% | 4.1% | 0.4% |
| orientation_penalty | -0.008153 | -0.7% | 0.7% | 2.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
