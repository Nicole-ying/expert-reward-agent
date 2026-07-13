# Training Feedback

## Final-policy outcome
score=27.344716, len=800.400000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-72.852135, 191.942581]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 202.433499 | 92.9% | 92.9% | 4.6% |
| goal_proximity | 9.190898 | 4.2% | 4.8% | 100.0% |
| velocity_penalty | -3.367883 | -1.5% | 1.5% | 100.0% |
| orientation_penalty | -1.665725 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
