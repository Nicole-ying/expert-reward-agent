# Training Feedback

## Final-policy outcome
score=304.881709, len=1154.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[303.725981, 306.125043]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 347.579201 | 83.9% | 84.0% | 100.0% |
| upright_penalty | -33.711951 | -8.1% | 8.1% | 100.0% |
| double_flight_penalty | -32.720000 | -7.9% | 7.9% | 14.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
