# Training Feedback

## Final-policy outcome
score=50.781573, len=632.750000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-169.364891, 234.434503]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 2492.000000 | 97.3% | 97.3% | 9.3% |
| proximity_improvement | 20.629822 | 0.8% | 1.9% | 99.9% |
| velocity_penalty | -16.625160 | -0.6% | 0.6% | 99.9% |
| angle_penalty | -4.031920 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
