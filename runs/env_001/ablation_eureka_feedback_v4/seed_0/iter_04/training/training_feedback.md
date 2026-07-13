# Training Feedback

## Final-policy outcome
score=46.215985, len=803.900000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-102.319976, 190.705899]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 794.754957 | 52.3% | 52.3% | 100.0% |
| progress | 536.596321 | 35.3% | 35.3% | 100.0% |
| contact | 187.500000 | 12.3% | 12.3% | 27.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
