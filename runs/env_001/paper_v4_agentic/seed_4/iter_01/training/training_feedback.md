# Training Feedback

## Final-policy outcome
score=-110.631593, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-126.893586, -93.288786]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 0.456958 | 75.5% | 75.5% | 0.7% |
| progress_gated | 0.148456 | 24.5% | 24.5% | 92.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
