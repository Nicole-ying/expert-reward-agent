# Training Feedback

## Final-policy outcome
score=-119.556687, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.709000, -97.923914]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_cost | -75.307647 | -84.3% | 84.3% | 100.0% |
| stability_cost | -10.031606 | -11.2% | 11.2% | 100.0% |
| soft_landing_bonus | 4.000000 | 4.5% | 4.5% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
