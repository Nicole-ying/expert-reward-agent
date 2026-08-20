# Training Feedback

## Final-policy outcome
score=-98.886267, len=752.800000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-207.682766, 61.177651]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_landing_reward | 6315.711812 | 93.6% | 93.6% | 100.0% |
| distance_cost | -426.013595 | -6.3% | 6.3% | 100.0% |
| stability_cost | -6.479766 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
