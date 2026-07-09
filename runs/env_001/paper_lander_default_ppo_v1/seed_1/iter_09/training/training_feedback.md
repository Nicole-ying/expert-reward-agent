# Training Feedback

## Final-policy outcome
score=-108.868629, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.445561, -89.487943]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_tracking_reward | -42.507739 | -73.8% | 73.8% | 100.0% |
| landing_improvement | 11.709494 | 20.3% | 24.2% | 100.0% |
| progress_reward | 1.120421 | 1.9% | 2.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
