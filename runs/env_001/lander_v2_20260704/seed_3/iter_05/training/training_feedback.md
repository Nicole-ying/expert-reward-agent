# Training Feedback

## Final-policy outcome
score=-236.148516, len=835.500000, terminated=5/20, truncated=15/20, reward_errors=0
score_range=[-297.616417, -144.606198]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| shaping_reward | 894.241736 | 100.0% | 100.0% | 100.0% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
