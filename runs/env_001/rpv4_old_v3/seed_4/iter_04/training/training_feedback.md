# Training Feedback

## Final-policy outcome
score=24.688193, len=754.050000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-84.406107, 209.988146]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pos | 7.914258 | 42.5% | 58.1% | 100.0% |
| contact | 5.368801 | 28.8% | 28.8% | 0.5% |
| vel | -1.681669 | -9.0% | 9.0% | 100.0% |
| angle | -0.454809 | -2.4% | 2.4% | 100.0% |
| angvel | -0.311669 | -1.7% | 1.7% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
