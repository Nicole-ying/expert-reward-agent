# Training Feedback

## Final-policy outcome
score=-36.964653, len=255.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-81.428413, 80.263518]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 105.218582 | 96.7% | 97.1% | 99.6% |
| action_efficiency_penalty | -3.168688 | -2.9% | 2.9% | 100.0% |
| angular_velocity_penalty | -0.036026 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 6/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
