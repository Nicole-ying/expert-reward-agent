# Training Feedback

## Final-policy outcome
score=-52.263827, len=297.450000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-84.132203, -16.431459]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vertical_gate | 288.574986 | 34.4% | 34.4% | 100.0% |
| angle_gate | 235.291199 | 28.1% | 28.1% | 99.6% |
| health_gate | 228.253058 | 27.2% | 27.2% | 99.6% |
| progress_raw | 84.658144 | 10.1% | 10.2% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
