# Training Feedback

## Final-policy outcome
score=-75.447203, len=694.400000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-1893.452083, 759.292384]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 924.954279 | 86.4% | 86.4% | 70.7% |
| lateral_drift_hinge | -88.576924 | -8.3% | 8.3% | 50.4% |
| roll_pitch_hinge | -57.586160 | -5.4% | 5.4% | 44.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
