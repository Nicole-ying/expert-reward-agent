# Training Feedback

## Final-policy outcome
score=-40.037266, len=154.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-133.278295, 15.315072]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 2.138831 | 80.7% | 86.4% | 100.0% |
| living_penalty | -0.308100 | -11.6% | 11.6% | 100.0% |
| soft_landing_penalty | -0.052633 | -2.0% | 2.0% | 1.5% |
| crash_prevention_x | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_prevention_y | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
