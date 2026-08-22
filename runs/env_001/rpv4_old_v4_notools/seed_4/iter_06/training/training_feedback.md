# Training Feedback

## Final-policy outcome
score=-111.912905, len=69.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-196.679836, 17.571949]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 55.745291 | 66.5% | 69.0% | 100.0% |
| attitude | -14.950602 | -17.8% | 17.8% | 100.0% |
| landing | -5.677077 | -6.8% | 12.4% | 0.6% |
| fuel | -0.610000 | -0.7% | 0.7% | 6.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
