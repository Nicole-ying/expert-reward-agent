# Training Feedback

## Final-policy outcome
score=-63.340537, len=217.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-89.731580, -23.407339]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 59.710870 | 96.1% | 96.1% | 95.4% |
| action_penalty | -2.418903 | -3.9% | 3.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
