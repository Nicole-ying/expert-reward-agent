# Training Feedback

## Final-policy outcome
score=200.987626, len=626.050000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[54.033136, 275.481878]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 316.337239 | 97.8% | 97.8% | 51.0% |
| approach_velocity_penalty | -5.152263 | -1.6% | 1.6% | 99.9% |
| progress | 1.302700 | 0.4% | 0.4% | 100.0% |
| orientation_penalty | -0.678321 | -0.2% | 0.2% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
