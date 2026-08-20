# Training Feedback

## Final-policy outcome
score=-105.533347, len=71.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-176.166702, 17.572554]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 0.574975 | 39.5% | 39.6% | 100.0% |
| shaped_progress | 0.499268 | 34.3% | 34.4% | 100.0% |
| landing_contact_reward | 0.288652 | 19.8% | 19.8% | 5.3% |
| angle_hinge_penalty | -0.053415 | -3.7% | 3.7% | 12.1% |
| action_cost | -0.036500 | -2.5% | 2.5% | 5.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
