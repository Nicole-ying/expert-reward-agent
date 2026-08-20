# Training Feedback

## Final-policy outcome
score=-115.206637, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.247427, -100.640778]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success | 25.000000 | 53.3% | 53.3% | 0.4% |
| soft_landing | -20.743517 | -44.2% | 44.2% | 100.0% |
| goal_approach | 1.118841 | 2.4% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
