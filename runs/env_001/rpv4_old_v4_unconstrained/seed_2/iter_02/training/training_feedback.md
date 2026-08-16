# Training Feedback

## Final-policy outcome
score=252.826018, len=386.650000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[179.071889, 294.371685]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 50.122683 | 70.0% | 70.0% | 26.7% |
| velocity_management | -13.804043 | -19.3% | 19.3% | 98.3% |
| progress | 4.149977 | 5.8% | 5.9% | 96.3% |
| orientation_penalty | -3.462125 | -4.8% | 4.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
