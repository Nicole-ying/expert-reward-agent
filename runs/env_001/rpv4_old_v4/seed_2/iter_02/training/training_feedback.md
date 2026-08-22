# Training Feedback

## Final-policy outcome
score=157.074640, len=925.850000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[108.064116, 264.128283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 155.533046 | 2.8% | 99.9% | 81.0% |
| progress | 2.587737 | 0.0% | 0.1% | 100.0% |
| orientation_penalty | -1.458872 | -0.0% | 0.0% | 100.0% |
| velocity_penalty | -0.209426 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
