# Training Feedback

## Final-policy outcome
score=107.144545, len=923.300000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[46.096323, 216.897068]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| position_proximity | 745.155420 | 50.2% | 50.2% | 100.0% |
| contact_completion | 736.822215 | 49.6% | 49.6% | 100.0% |
| stable_orientation | -1.723192 | -0.1% | 0.1% | 100.0% |
| soft_landing_velocity | -0.408948 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
