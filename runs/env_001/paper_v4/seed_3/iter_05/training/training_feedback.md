# Training Feedback

## Final-policy outcome
score=249.342342, len=374.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[216.589059, 285.568998]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 33.129554 | 52.3% | 52.3% | 10.9% |
| engine_penalty | -17.027500 | -26.9% | 26.9% | 90.8% |
| progress | 6.909746 | 10.9% | 11.1% | 96.2% |
| velocity_penalty | -3.604593 | -5.7% | 5.7% | 96.8% |
| orientation_penalty | -2.522049 | -4.0% | 4.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
