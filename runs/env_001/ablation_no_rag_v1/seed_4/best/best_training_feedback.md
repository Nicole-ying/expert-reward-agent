# Training Feedback

## Final-policy outcome
score=240.862059, len=468.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[209.162689, 267.880162]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 197.800000 | 94.8% | 94.8% | 42.2% |
| target_proximity | -8.407849 | -4.0% | 4.0% | 100.0% |
| velocity_penalty | -1.723487 | -0.8% | 0.8% | 97.9% |
| engine_penalty | -0.443550 | -0.2% | 0.2% | 94.7% |
| orientation_penalty | -0.163757 | -0.1% | 0.1% | 100.0% |
| angvel_penalty | -0.093907 | -0.0% | 0.0% | 62.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
