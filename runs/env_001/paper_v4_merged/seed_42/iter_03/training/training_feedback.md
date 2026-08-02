# Training Feedback

## Final-policy outcome
score=260.060104, len=316.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[233.959286, 296.266506]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_bonus | 89.018430 | 62.8% | 62.8% | 60.9% |
| angular_stability | 24.841976 | 17.5% | 17.5% | 97.8% |
| approach_reward | 23.630877 | 16.7% | 16.7% | 100.0% |
| fuel_penalty | -2.844000 | -2.0% | 2.0% | 89.9% |
| goal_progress | 1.375708 | 1.0% | 1.0% | 95.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
