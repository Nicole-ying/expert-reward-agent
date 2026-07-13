# Training Feedback

## Final-policy outcome
score=-17.897087, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-45.375937, 21.479454]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| position_proximity | 749.311080 | 99.7% | 99.7% | 100.0% |
| stable_orientation | -1.720382 | -0.2% | 0.2% | 100.0% |
| soft_landing_velocity | -0.332431 | -0.0% | 0.0% | 100.0% |
| contact_completion | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
