# Training Feedback

## Final-policy outcome
score=-14.181841, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-47.819839, 15.213899]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 993.556076 | 55.5% | 55.5% | 100.0% |
| progress | 796.597307 | 44.5% | 44.5% | 100.0% |
| contact | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
