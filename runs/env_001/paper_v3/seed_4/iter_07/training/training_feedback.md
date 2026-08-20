# Training Feedback

## Final-policy outcome
score=-34.288724, len=980.550000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-100.945280, 24.871863]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| descent_quality | 96.676468 | 64.2% | 64.2% | 100.0% |
| proximity | 36.745226 | 24.4% | 35.3% | 100.0% |
| contact_quality | 0.449530 | 0.3% | 0.3% | 0.0% |
| velocity_penalty | -0.212446 | -0.1% | 0.1% | 100.0% |
| attitude_penalty | -0.060002 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
