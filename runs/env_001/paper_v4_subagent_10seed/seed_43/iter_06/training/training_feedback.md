# Training Feedback

## Final-policy outcome
score=-117.778715, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.853315, -98.092066]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 0.923771 | 64.2% | 66.0% | 100.0% |
| landing_soft_reward | 0.413275 | 28.7% | 28.7% | 0.7% |
| angle_penalty | -0.048717 | -3.4% | 3.4% | 100.0% |
| action_cost | -0.027500 | -1.9% | 1.9% | 4.0% |
| boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
