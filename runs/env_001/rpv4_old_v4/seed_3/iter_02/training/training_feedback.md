# Training Feedback

## Final-policy outcome
score=107.129410, len=775.600000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-33.715767, 264.822182]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 154.464574 | 92.5% | 92.5% | 2.8% |
| velocity_penalty | -5.313230 | -3.2% | 3.2% | 99.2% |
| goal_proximity | 3.962801 | 2.4% | 2.5% | 99.1% |
| orientation_penalty | -3.090273 | -1.9% | 1.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
