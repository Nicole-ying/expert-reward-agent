# Training Feedback

## Final-policy outcome
score=26.735066, len=341.750000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-78.024087, 258.305735]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 555.994366 | 85.4% | 85.4% | 31.9% |
| pos | -88.133305 | -13.5% | 13.5% | 100.0% |
| vel | -5.300010 | -0.8% | 0.8% | 100.0% |
| angvel | -1.237484 | -0.2% | 0.2% | 92.7% |
| angle | -0.444999 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
