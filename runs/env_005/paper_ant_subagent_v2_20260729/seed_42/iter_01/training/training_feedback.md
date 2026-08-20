# Training Feedback

## Final-policy outcome
score=-336.844660, len=944.400000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-1493.024366, 77.836696]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 853.951211 | 86.8% | 86.8% | 67.2% |
| roll_pitch_hinge | -66.447967 | -6.8% | 6.8% | 39.0% |
| lateral_drift_hinge | -63.782634 | -6.5% | 6.5% | 39.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
