# Training Feedback

## Final-policy outcome
score=-15.733251, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-41.618148, 19.633448]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 7.860524 | 36.8% | 39.1% | 100.0% |
| velocity_penalty | -5.724752 | -26.8% | 26.8% | 100.0% |
| orientation_penalty | -3.916764 | -18.3% | 18.3% | 100.0% |
| descent_incentive | 3.366429 | 15.8% | 15.8% | 73.8% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
