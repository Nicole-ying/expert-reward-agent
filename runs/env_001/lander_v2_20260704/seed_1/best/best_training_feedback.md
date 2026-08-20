# Training Feedback

## Final-policy outcome
score=134.647832, len=459.500000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-149.223591, 283.562155]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_landing_reward | 792.964804 | 96.7% | 96.7% | 94.2% |
| distance_cost | -14.163082 | -1.7% | 1.7% | 100.0% |
| stability_cost | -12.923618 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
