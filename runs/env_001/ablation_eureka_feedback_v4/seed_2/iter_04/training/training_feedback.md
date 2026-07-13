# Training Feedback

## Final-policy outcome
score=-122.562724, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -98.057243]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| vertical_penalty | -1.219265 | -47.6% | 47.6% | 74.4% |
| progress | 1.115021 | 43.6% | 45.1% | 100.0% |
| contact_bonus | 0.152580 | 6.0% | 6.0% | 3.1% |
| fuel_cost | -0.032000 | -1.3% | 1.3% | 2.3% |
| angle_penalty | -0.001437 | -0.1% | 0.1% | 2.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
