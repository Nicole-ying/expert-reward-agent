# Training Feedback

## Final-policy outcome
score=3.039331, len=965.950000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-51.586939, 201.807159]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_envelope | 1670.925091 | 98.8% | 98.8% | 100.0% |
| velocity_penalty | -9.010700 | -0.5% | 0.5% | 100.0% |
| goal_proximity | 5.029909 | 0.3% | 0.4% | 100.0% |
| upright_penalty | -5.312046 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
