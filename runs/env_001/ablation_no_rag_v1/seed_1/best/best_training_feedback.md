# Training Feedback

## Final-policy outcome
score=128.952616, len=682.600000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-186.039911, 273.130754]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_guidance | 13247.748580 | 99.9% | 100.0% | 77.7% |
| tilt_penalty | -3.656006 | -0.0% | 0.0% | 100.0% |
| distance_progress | 2.361488 | 0.0% | 0.0% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
