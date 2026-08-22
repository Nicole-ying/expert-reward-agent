# Training Feedback

## Final-policy outcome
score=170.641744, len=363.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.179203, 261.670162]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 38.508361 | 45.6% | 45.6% | 12.3% |
| progress | 21.076008 | 24.9% | 32.7% | 97.3% |
| landing_velocity_penalty | -6.959595 | -8.2% | 8.2% | 13.9% |
| attitude_penalty | -6.652874 | -7.9% | 7.9% | 100.0% |
| engine_cost | -4.804000 | -5.7% | 5.7% | 66.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
