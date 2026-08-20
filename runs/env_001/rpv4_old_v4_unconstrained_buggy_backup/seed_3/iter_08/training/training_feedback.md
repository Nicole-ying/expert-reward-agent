# Training Feedback

## Final-policy outcome
score=154.328659, len=415.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-62.691647, 260.370380]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 3695.250000 | 99.7% | 99.7% | 13.7% |
| velocity_penalty | -6.637381 | -0.2% | 0.2% | 98.1% |
| goal_proximity | 2.296536 | 0.1% | 0.1% | 98.0% |
| orientation_penalty | -2.097178 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
