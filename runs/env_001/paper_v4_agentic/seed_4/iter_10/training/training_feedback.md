# Training Feedback

## Final-policy outcome
score=-36.360283, len=974.850000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-176.324300, 12.276626]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing | 529.532846 | 98.3% | 98.3% | 99.7% |
| progress_delta | 7.970377 | 1.5% | 1.5% | 64.0% |
| brake_reward | 1.364208 | 0.3% | 0.3% | 3.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
