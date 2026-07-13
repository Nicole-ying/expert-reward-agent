# Training Feedback

## Final-policy outcome
score=192.081449, len=537.050000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[25.538657, 264.564248]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| descent_quality | 1111.605769 | 73.0% | 73.0% | 100.0% |
| proximity | 410.983088 | 27.0% | 27.0% | 100.0% |
| velocity_penalty | -0.304686 | -0.0% | 0.0% | 98.3% |
| attitude_penalty | -0.074306 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
