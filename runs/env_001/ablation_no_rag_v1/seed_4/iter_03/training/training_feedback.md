# Training Feedback

## Final-policy outcome
score=102.161665, len=413.800000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-126.856085, 246.237834]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 146.300000 | 76.5% | 76.5% | 35.4% |
| target_proximity | -40.281333 | -21.1% | 21.1% | 100.0% |
| velocity_penalty | -3.762735 | -2.0% | 2.0% | 99.7% |
| orientation_penalty | -0.620570 | -0.3% | 0.3% | 100.0% |
| angvel_penalty | -0.294685 | -0.2% | 0.2% | 98.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
