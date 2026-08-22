# Training Feedback

## Final-policy outcome
score=-183.316556, len=947.350000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-284.463711, -99.250079]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_proxy | 5792.231178 | 60.1% | 60.1% | 100.0% |
| goal_proximity | -3768.392629 | -39.1% | 39.1% | 100.0% |
| thrust_penalty | -47.367500 | -0.5% | 0.5% | 100.0% |
| velocity_penalty | -24.149682 | -0.3% | 0.3% | 100.0% |
| angle_penalty | -9.949903 | -0.1% | 0.1% | 100.0% |
| contact_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
