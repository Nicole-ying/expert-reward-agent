# Training Feedback

## Final-policy outcome
score=-388.820358, len=702.300000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-1005.673116, -12.933700]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 640.835676 | 81.7% | 83.3% | 93.6% |
| angular_penalty | -118.070765 | -15.1% | 15.1% | 94.9% |
| lateral_penalty | -13.081515 | -1.7% | 1.7% | 94.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
