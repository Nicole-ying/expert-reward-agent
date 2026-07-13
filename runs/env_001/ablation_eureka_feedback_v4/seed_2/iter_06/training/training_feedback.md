# Training Feedback

## Final-policy outcome
score=-122.023680, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.398763, -98.057243]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.115648 | 56.4% | 58.4% | 100.0% |
| vertical_penalty | -0.634408 | -32.1% | 32.1% | 74.2% |
| contact_bonus | 0.155433 | 7.9% | 7.9% | 3.0% |
| fuel_cost | -0.033000 | -1.7% | 1.7% | 2.4% |
| angle_penalty | -0.000564 | -0.0% | 0.0% | 1.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
