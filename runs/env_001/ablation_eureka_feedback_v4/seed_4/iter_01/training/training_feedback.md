# Training Feedback

## Final-policy outcome
score=-19.324311, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-48.476401, 11.024084]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -65.224731 | -94.3% | 94.3% | 9.1% |
| approach_progress | 2.382989 | 3.4% | 3.7% | 100.0% |
| angle_stability | -1.413554 | -2.0% | 2.0% | 100.0% |
| landing_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
