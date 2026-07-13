# Training Feedback

## Final-policy outcome
score=-298.173850, len=79.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-395.995720, -202.486050]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.886592 | 53.8% | 58.4% | 100.0% |
| fuel_cost | -0.346000 | -21.0% | 21.0% | 21.7% |
| soft_landing | -0.338635 | -20.6% | 20.6% | 6.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
