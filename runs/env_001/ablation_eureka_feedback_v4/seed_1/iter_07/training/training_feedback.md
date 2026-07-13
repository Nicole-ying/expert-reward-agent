# Training Feedback

## Final-policy outcome
score=-115.040584, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582367, -97.000369]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | -33.259243 | -79.2% | 79.2% | 100.0% |
| success_reward | 4.465227 | 10.6% | 10.6% | 1.8% |
| velocity_penalty | -2.480899 | -5.9% | 5.9% | 99.8% |
| orientation_penalty | -1.765679 | -4.2% | 4.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
