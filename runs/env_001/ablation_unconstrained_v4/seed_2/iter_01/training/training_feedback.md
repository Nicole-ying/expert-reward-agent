# Training Feedback

## Final-policy outcome
score=-111.843325, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.705727, -78.933292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -73.933146 | -84.0% | 84.0% | 100.0% |
| soft_landing_stabilization | -9.302658 | -10.6% | 10.6% | 99.8% |
| contact_establishment | 3.400000 | 3.9% | 3.9% | 3.2% |
| upright_attitude | -1.394754 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
