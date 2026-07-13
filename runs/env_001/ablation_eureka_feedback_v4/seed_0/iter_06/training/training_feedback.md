# Training Feedback

## Final-policy outcome
score=110.889487, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[82.383322, 143.249402]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 994.011586 | 76.9% | 76.9% | 100.0% |
| contact | 290.450000 | 22.5% | 22.5% | 43.1% |
| progress | 7.628805 | 0.6% | 0.6% | 69.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
