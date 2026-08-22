# Training Feedback

## Final-policy outcome
score=-114.322233, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.122909, -92.382728]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 33.594475 | 69.2% | 71.6% | 100.0% |
| descent_bonus | 7.172625 | 14.8% | 14.8% | 94.3% |
| soft_landing | 4.031075 | 8.3% | 8.3% | 1.6% |
| attitude_penalty | -2.079212 | -4.3% | 4.3% | 100.0% |
| fuel_cost | -0.490000 | -1.0% | 1.0% | 7.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
