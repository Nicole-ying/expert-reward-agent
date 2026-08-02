# Training Feedback

## Final-policy outcome
score=157.855989, len=933.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[117.465240, 274.283355]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 652.471025 | 97.8% | 97.8% | 73.3% |
| progress | 13.738687 | 2.1% | 2.1% | 99.8% |
| angle_penalty | -0.330834 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.095876 | -0.0% | 0.0% | 66.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
