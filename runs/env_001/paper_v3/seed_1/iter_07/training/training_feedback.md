# Training Feedback

## Final-policy outcome
score=-7.765426, len=972.450000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-48.616794, 121.034979]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -116.866286 | -83.4% | 83.4% | 100.0% |
| contact_reward | 15.017095 | 10.7% | 10.7% | 1.1% |
| velocity_penalty | -7.285143 | -5.2% | 5.2% | 99.9% |
| orientation_penalty | -0.966535 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
