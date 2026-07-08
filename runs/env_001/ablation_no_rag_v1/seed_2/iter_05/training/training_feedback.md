# Training Feedback

## Final-policy outcome
score=-24.456569, len=396.350000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-171.806166, 234.293864]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| angle_penalty | -4.163748 | -55.6% | 55.6% | 100.0% |
| potential_diff | 0.881320 | 11.8% | 44.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
