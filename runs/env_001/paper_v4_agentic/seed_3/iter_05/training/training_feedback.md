# Training Feedback

## Final-policy outcome
score=241.723383, len=443.450000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[139.580747, 291.345925]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 50.283116 | 56.6% | 56.6% | 23.4% |
| progress | 27.652928 | 31.1% | 31.8% | 97.2% |
| engine_cost | -7.417000 | -8.3% | 8.3% | 83.6% |
| attitude_penalty | -2.431167 | -2.7% | 2.7% | 100.0% |
| landing_velocity_penalty | -0.431218 | -0.5% | 0.5% | 21.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
