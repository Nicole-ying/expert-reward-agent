# Training Feedback

## Final-policy outcome
score=201.427675, len=797.900000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[130.762772, 290.911365]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 133.230775 | 97.2% | 97.2% | 70.1% |
| progress | 2.523319 | 1.8% | 1.9% | 99.3% |
| orientation_penalty | -1.115411 | -0.8% | 0.8% | 100.0% |
| velocity_penalty | -0.177857 | -0.1% | 0.1% | 99.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
