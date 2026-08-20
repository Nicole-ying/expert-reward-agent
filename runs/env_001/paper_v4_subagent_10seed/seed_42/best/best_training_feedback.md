# Training Feedback

## Final-policy outcome
score=194.629572, len=630.200000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[98.753115, 248.714668]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 473.485526 | 89.7% | 89.7% | 100.0% |
| soft_landing | 54.232009 | 10.3% | 10.3% | 9.7% |
| orientation_penalty | -0.048327 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
