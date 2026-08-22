# Training Feedback

## Final-policy outcome
score=-55.536253, len=724.550000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-791.335081, 227.612256]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated_height | 796.229438 | 94.8% | 96.2% | 78.6% |
| action_penalty | -32.076949 | -3.8% | 3.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
