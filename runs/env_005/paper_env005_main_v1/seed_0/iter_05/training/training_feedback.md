# Training Feedback

## Final-policy outcome
score=11.045470, len=252.350000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-774.907723, 230.516343]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 247.510956 | 62.7% | 67.6% | 100.0% |
| orientation_reward | 44.995392 | 11.4% | 18.5% | 100.0% |
| lateral_velocity_penalty | -28.990283 | -7.3% | 7.3% | 100.0% |
| angular_velocity_penalty | -25.859236 | -6.6% | 6.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
