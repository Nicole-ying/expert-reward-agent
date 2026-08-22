# Training Feedback

## Final-policy outcome
score=-166.291060, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-199.706985, -124.135086]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_bonus | 768.778240 | 94.3% | 94.3% | 100.0% |
| action_reward | -28.115500 | -3.4% | 5.3% | 100.0% |
| progress_reward | -0.723104 | -0.1% | 0.1% | 100.0% |
| vel_damping | -1.107418 | -0.1% | 0.1% | 3.8% |
| orientation_penalty | -0.831768 | -0.1% | 0.1% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
