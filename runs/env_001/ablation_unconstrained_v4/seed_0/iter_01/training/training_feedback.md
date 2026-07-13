# Training Feedback

## Final-policy outcome
score=-121.659055, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -97.743423]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_to_goal | -66.538449 | -95.2% | 95.2% | 100.0% |
| successful_settle_proxy | 1.903372 | 2.7% | 2.7% | 3.1% |
| orientation_penalty | -1.231780 | -1.8% | 1.8% | 100.0% |
| engine_efficiency | -0.190000 | -0.3% | 0.3% | 2.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
