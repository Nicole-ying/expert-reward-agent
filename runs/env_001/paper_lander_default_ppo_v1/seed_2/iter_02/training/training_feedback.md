# Training Feedback

## Final-policy outcome
score=-109.878946, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.380824, -85.998511]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -9.678942 | -79.0% | 79.0% | 100.0% |
| distance_progress | 2.242088 | 18.3% | 19.0% | 100.0% |
| soft_landing_proxy | 0.250000 | 2.0% | 2.0% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
