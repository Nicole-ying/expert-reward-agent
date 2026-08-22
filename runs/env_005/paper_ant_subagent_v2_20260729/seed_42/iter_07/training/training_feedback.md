# Training Feedback

## Final-policy outcome
score=-415.211675, len=299.050000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-1702.611675, 18.873805]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 74.738484 | 38.8% | 72.1% | 75.5% |
| lateral_penalty | -17.182856 | -8.9% | 8.9% | 21.6% |
| roll_penalty | -15.075047 | -7.8% | 7.8% | 21.1% |
| pitch_penalty | -12.114540 | -6.3% | 6.3% | 19.1% |
| height_low_penalty | -8.964256 | -4.7% | 4.7% | 74.7% |
| height_high_penalty | -0.301906 | -0.2% | 0.2% | 1.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
