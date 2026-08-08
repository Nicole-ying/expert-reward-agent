# Training Feedback

## Final-policy outcome
score=58.862720, len=559.900000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-211.881499, 215.423021]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_contact_reward | 292.891768 | 84.2% | 84.2% | 5.6% |
| progress_reward | -50.796714 | -14.6% | 14.6% | 99.9% |
| orientation_penalty | -3.967178 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
