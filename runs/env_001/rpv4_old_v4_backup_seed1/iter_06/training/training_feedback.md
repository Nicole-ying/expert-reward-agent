# Training Feedback

## Final-policy outcome
score=-81.797020, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-149.216081, 84.983972]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | -3.075439 | -29.4% | 29.4% | 100.0% |
| goal_proximity | 1.774610 | 17.0% | 25.7% | 100.0% |
| velocity_penalty | -2.385383 | -22.8% | 22.8% | 100.0% |
| landing_approach | 0.974175 | 9.3% | 22.1% | 98.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
