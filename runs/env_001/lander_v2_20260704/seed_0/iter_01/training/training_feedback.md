# Training Feedback

## Final-policy outcome
score=-518.689308, len=62.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-683.540428, -313.351760]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -62.707678 | -96.2% | 96.2% | 100.0% |
| stability_penalty | -2.497156 | -3.8% | 3.8% | 100.0% |
| soft_landing_proxy | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
