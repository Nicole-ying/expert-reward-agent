# Training Feedback

## Final-policy outcome
score=3.546072, len=828.350000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-193.772208, 189.322250]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 91.892725 | 83.5% | 83.5% | 2.7% |
| pos | 5.408563 | 4.9% | 8.6% | 99.8% |
| vel | -7.224800 | -6.6% | 6.6% | 99.7% |
| angle | -0.823012 | -0.7% | 0.7% | 100.0% |
| angvel | -0.708924 | -0.6% | 0.6% | 99.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
