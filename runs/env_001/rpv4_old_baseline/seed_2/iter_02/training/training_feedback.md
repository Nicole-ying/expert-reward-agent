# Training Feedback

## Final-policy outcome
score=-64.687764, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-95.465184, -28.546546]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| angle_reward | 798.124172 | 52.0% | 52.0% | 100.0% |
| speed_reward | 690.653978 | 45.0% | 45.0% | 100.0% |
| proximity_reward | -47.047516 | -3.1% | 3.1% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
