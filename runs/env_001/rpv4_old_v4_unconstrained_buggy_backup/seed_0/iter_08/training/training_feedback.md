# Training Feedback

## Final-policy outcome
score=-37.654175, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-66.645362, 5.405622]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_approach_reward | 1691.885359 | 91.9% | 91.9% | 100.0% |
| action_penalty | -136.167500 | -7.4% | 7.4% | 100.0% |
| progress_reward | 12.339326 | 0.7% | 0.7% | 100.0% |
| orientation_penalty | -0.624208 | -0.0% | 0.0% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
