# Training Feedback

## Final-policy outcome
score=144.304309, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[101.342861, 171.587967]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 13.960004 | 56.7% | 60.3% | 100.0% |
| landing_reward | 8.034063 | 32.7% | 32.7% | 98.5% |
| angle_penalty | -1.697453 | -6.9% | 6.9% | 100.0% |
| speed_penalty | -0.042040 | -0.2% | 0.2% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
