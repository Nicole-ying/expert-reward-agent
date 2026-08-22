# Training Feedback

## Final-policy outcome
score=-120.540069, len=772.100000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-235.073025, 61.411173]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 32.000000 | 64.0% | 64.0% | 0.1% |
| goal_proximity | 3.659826 | 7.3% | 16.6% | 100.0% |
| velocity_penalty | -7.418051 | -14.8% | 14.8% | 100.0% |
| orientation_penalty | -2.301179 | -4.6% | 4.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
