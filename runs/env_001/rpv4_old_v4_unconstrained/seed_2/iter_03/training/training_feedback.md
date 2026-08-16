# Training Feedback

## Final-policy outcome
score=130.171654, len=993.400000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[92.770420, 237.389735]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 236.962345 | 95.4% | 95.4% | 69.0% |
| progress | 6.504186 | 2.6% | 3.1% | 100.0% |
| orientation_penalty | -3.807491 | -1.5% | 1.5% | 100.0% |
| velocity_management | -0.093068 | -0.0% | 0.0% | 13.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
