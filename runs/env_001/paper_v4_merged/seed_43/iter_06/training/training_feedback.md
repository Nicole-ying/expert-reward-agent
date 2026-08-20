# Training Feedback

## Final-policy outcome
score=-114.350062, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.849800, -89.942949]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.105549 | 45.9% | 49.1% | 100.0% |
| shaped_progress | 0.911427 | 37.8% | 40.0% | 100.0% |
| landing_contact_reward | 0.208765 | 8.7% | 8.7% | 3.0% |
| action_cost | -0.055000 | -2.3% | 2.3% | 8.0% |
| angle_hinge_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
