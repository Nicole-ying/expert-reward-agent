# Training Feedback

## Final-policy outcome
score=130.237220, len=468.300000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-84.388719, 266.860159]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_landing_reward | 1127.308915 | 98.1% | 98.1% | 97.4% |
| distance_cost | -11.343280 | -1.0% | 1.0% | 100.0% |
| stability_cost | -9.919037 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
