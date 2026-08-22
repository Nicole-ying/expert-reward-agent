# Training Feedback

## Final-policy outcome
score=14.435817, len=759.900000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-200.905569, 289.194894]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_progress | 24.929501 | 26.7% | 54.9% | 99.9% |
| soft_landing_bonus | 39.133277 | 41.9% | 41.9% | 3.4% |
| attitude_penalty | -2.983197 | -3.2% | 3.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
