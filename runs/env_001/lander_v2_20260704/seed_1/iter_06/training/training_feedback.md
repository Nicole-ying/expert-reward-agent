# Training Feedback

## Final-policy outcome
score=-125.742018, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -84.439466]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_landing_reward | 39.801803 | 70.2% | 70.2% | 99.7% |
| stability_cost | -10.633001 | -18.7% | 18.7% | 100.0% |
| distance_cost | -6.032561 | -10.6% | 10.6% | 100.0% |
| engine_penalty | -0.270000 | -0.5% | 0.5% | 1.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
