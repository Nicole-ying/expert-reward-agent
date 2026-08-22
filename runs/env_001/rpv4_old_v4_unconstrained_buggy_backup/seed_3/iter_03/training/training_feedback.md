# Training Feedback

## Final-policy outcome
score=-50.454204, len=754.300000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-210.441144, 205.911676]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 35.000000 | 84.6% | 84.6% | 0.0% |
| velocity_penalty | -3.533354 | -8.5% | 8.5% | 99.9% |
| orientation_penalty | -1.511864 | -3.7% | 3.7% | 100.0% |
| goal_proximity | 1.015898 | 2.5% | 3.2% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
