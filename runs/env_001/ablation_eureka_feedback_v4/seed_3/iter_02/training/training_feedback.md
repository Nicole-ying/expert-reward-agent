# Training Feedback

## Final-policy outcome
score=115.514506, len=725.700000, terminated=12/20, truncated=8/20, reward_errors=0
score_range=[-40.353873, 206.271101]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | 544.782466 | 51.5% | 51.5% | 100.0% |
| proximity | 507.197093 | 47.9% | 47.9% | 100.0% |
| energy_penalty | -6.231000 | -0.6% | 0.6% | 85.9% |
| terminal_velocity_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
