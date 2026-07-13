# Training Feedback

## Final-policy outcome
score=-129.324842, len=69.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-469.057036, 10.136052]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 5.707312 | 92.8% | 95.9% | 100.0% |
| fuel_penalty | -0.137500 | -2.2% | 2.2% | 3.9% |
| landing_reward | 0.114008 | 1.9% | 1.9% | 3.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
