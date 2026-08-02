# Training Feedback

## Final-policy outcome
score=-122.171546, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-146.822750, -101.571689]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.060033 | 77.5% | 83.2% | 100.0% |
| landing_contact_reward | 0.197708 | 14.5% | 14.5% | 3.1% |
| action_cost | -0.032000 | -2.3% | 2.3% | 4.7% |
| angle_hinge | -0.000076 | -0.0% | 0.0% | 0.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
