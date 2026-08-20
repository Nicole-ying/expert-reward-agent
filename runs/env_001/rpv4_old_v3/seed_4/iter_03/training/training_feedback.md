# Training Feedback

## Final-policy outcome
score=-21.395754, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-48.679777, 18.109703]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pos | 59.087881 | 84.7% | 97.9% | 100.0% |
| vel | -1.081692 | -1.6% | 1.6% | 100.0% |
| angle | -0.235360 | -0.3% | 0.3% | 100.0% |
| angvel | -0.182728 | -0.3% | 0.3% | 100.0% |
| contact | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
