# Training Feedback

## Final-policy outcome
score=-40.666010, len=991.500000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-134.657520, 7.001501]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pos | 8.455809 | 63.6% | 72.5% | 100.0% |
| contact | 2.287864 | 17.2% | 17.2% | 0.1% |
| vel | -0.903246 | -6.8% | 6.8% | 100.0% |
| angvel | -0.239863 | -1.8% | 1.8% | 100.0% |
| angle | -0.227131 | -1.7% | 1.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
