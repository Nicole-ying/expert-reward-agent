# Training Feedback

## Final-policy outcome
score=108.445395, len=629.450000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-175.789057, 249.167222]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_landing_reward | 606.663978 | 95.5% | 95.5% | 89.0% |
| stability_cost | -14.765450 | -2.3% | 2.3% | 100.0% |
| distance_cost | -14.082864 | -2.2% | 2.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
