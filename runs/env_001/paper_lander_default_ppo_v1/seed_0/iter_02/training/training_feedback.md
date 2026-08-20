# Training Feedback

## Final-policy outcome
score=-94.321457, len=73.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-136.905076, 65.985004]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.191987 | 76.0% | 78.5% | 100.0% |
| soft_landing_proxy | 0.225000 | 14.3% | 14.3% | 0.6% |
| stability_penalty | -0.112787 | -7.2% | 7.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
