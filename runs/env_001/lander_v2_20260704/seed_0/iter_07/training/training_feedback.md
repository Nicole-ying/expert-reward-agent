# Training Feedback

## Final-policy outcome
score=167.992192, len=647.400000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-41.151395, 248.794180]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 491.129187 | 88.1% | 88.1% | 100.0% |
| contact_reward | 51.210000 | 9.2% | 9.2% | 14.9% |
| velocity_penalty | -13.459726 | -2.4% | 2.4% | 99.9% |
| angle_penalty | -1.521084 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
