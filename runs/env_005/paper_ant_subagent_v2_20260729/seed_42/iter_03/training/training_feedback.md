# Training Feedback

## Final-policy outcome
score=-186.368149, len=752.350000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-2001.714754, 292.827301]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 849.391202 | 88.0% | 88.0% | 75.9% |
| lateral_drift_hinge | -64.338115 | -6.7% | 6.7% | 45.5% |
| roll_pitch_hinge | -47.278865 | -4.9% | 4.9% | 38.8% |
| height_boundary_hinge | -4.268929 | -0.4% | 0.4% | 17.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 4/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
