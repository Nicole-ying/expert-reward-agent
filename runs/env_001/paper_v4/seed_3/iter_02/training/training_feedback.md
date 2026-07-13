# Training Feedback

## Final-policy outcome
score=147.709303, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[116.945722, 180.823954]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 367.966735 | 95.9% | 95.9% | 76.2% |
| progress | 7.041011 | 1.8% | 1.9% | 100.0% |
| velocity_penalty | -4.473797 | -1.2% | 1.2% | 100.0% |
| orientation_penalty | -4.014124 | -1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
