# Training Feedback

## Final-policy outcome
score=-17.841914, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-47.289192, 21.597106]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 532.393852 | 98.2% | 98.2% | 78.8% |
| distance_reward | 6.109655 | 1.1% | 1.3% | 100.0% |
| stability_penalty | -2.238962 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
