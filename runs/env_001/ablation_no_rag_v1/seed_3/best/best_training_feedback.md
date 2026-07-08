# Training Feedback

## Final-policy outcome
score=214.808066, len=557.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[180.151594, 265.668850]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 72.100126 | 87.5% | 87.5% | 6.6% |
| progress_reward | 6.906196 | 8.4% | 8.8% | 97.4% |
| velocity_penalty | -2.848134 | -3.5% | 3.5% | 97.5% |
| attitude_penalty | -0.241169 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
