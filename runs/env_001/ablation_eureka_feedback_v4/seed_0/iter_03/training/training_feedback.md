# Training Feedback

## Final-policy outcome
score=-582.646088, len=65.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-744.291302, -420.950080]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | -65.891507 | -69.0% | 69.0% | 100.0% |
| stability | 28.462899 | 29.8% | 29.8% | 100.0% |
| contact | 1.075000 | 1.1% | 1.1% | 2.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
