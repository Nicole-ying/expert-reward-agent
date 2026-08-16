# Training Feedback

## Final-policy outcome
score=101.818742, len=786.150000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-59.358865, 248.273036]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 9810.275446 | 100.0% | 100.0% | 12.1% |
| orientation_penalty | -3.011177 | -0.0% | 0.0% | 100.0% |
| progress | 1.180675 | 0.0% | 0.0% | 100.0% |
| velocity_penalty | -0.228488 | -0.0% | 0.0% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
