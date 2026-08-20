# Training Feedback

## Final-policy outcome
score=57.502753, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[22.254096, 99.955954]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 89.464975 | 59.4% | 59.4% | 59.9% |
| upright_penalty | -42.816638 | -28.4% | 28.4% | 100.0% |
| velocity_penalty | -11.427432 | -7.6% | 7.6% | 100.0% |
| goal_proximity | 5.531467 | 3.7% | 4.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
