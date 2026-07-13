# Training Feedback

## Final-policy outcome
score=-125.271689, len=78.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-253.603378, 12.368148]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 1.618251 | 47.8% | 47.8% | 6.3% |
| progress | 1.237086 | 36.6% | 37.8% | 100.0% |
| fuel_cost | -0.487000 | -14.4% | 14.4% | 31.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
