# Training Feedback

## Final-policy outcome
score=220.242000, len=515.850000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[192.130706, 265.521557]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_completion | 724.455654 | 65.8% | 65.8% | 100.0% |
| position_proximity | 375.531294 | 34.1% | 34.1% | 100.0% |
| stable_orientation | -0.991143 | -0.1% | 0.1% | 100.0% |
| soft_landing_velocity | -0.436673 | -0.0% | 0.0% | 97.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
