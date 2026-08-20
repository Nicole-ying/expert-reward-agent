# Training Feedback

## Final-policy outcome
score=5.504575, len=968.500000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-1985.304120, 610.377687]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 1378.369901 | 88.5% | 88.5% | 82.7% |
| lateral_drift_hinge | -90.947874 | -5.8% | 5.8% | 49.2% |
| roll_pitch_hinge | -88.008512 | -5.7% | 5.7% | 52.6% |
| descent_early_penalty | -0.287754 | -0.0% | 0.0% | 9.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
