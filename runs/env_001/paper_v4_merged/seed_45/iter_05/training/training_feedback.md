# Training Feedback

## Final-policy outcome
score=186.993979, len=750.350000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[80.837909, 288.690092]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| completion_proxy | 517.813247 | 98.3% | 98.3% | 88.8% |
| progress_delta | 6.634889 | 1.3% | 1.3% | 99.3% |
| speed_penalty | -0.869971 | -0.2% | 0.2% | 8.0% |
| angle_penalty | -0.720982 | -0.1% | 0.1% | 4.2% |
| angvel_penalty | -0.057138 | -0.0% | 0.0% | 1.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
