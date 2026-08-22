# Training Feedback

## Final-policy outcome
score=235.033692, len=425.200000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[82.814065, 278.428241]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 219.677227 | 87.8% | 87.8% | 71.7% |
| proximity_progress | 26.805481 | 10.7% | 11.0% | 96.7% |
| attitude_penalty | -2.012444 | -0.8% | 0.8% | 100.0% |
| landing_speed | -1.047543 | -0.4% | 0.4% | 68.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
