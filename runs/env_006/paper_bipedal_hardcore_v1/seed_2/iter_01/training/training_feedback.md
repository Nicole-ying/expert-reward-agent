# Training Feedback

## Final-policy outcome
score=-59.968242, len=246.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-118.027902, -25.915322]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 77.784497 | 92.9% | 96.7% | 100.0% |
| balance_penalty | -2.724349 | -3.3% | 3.3% | 14.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
