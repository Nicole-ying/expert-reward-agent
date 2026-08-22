# Training Feedback

## Final-policy outcome
score=-119.410807, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.241873, -95.638888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 80.000000 | 73.0% | 73.0% | 0.6% |
| approach_reward | 11.159000 | 10.2% | 10.5% | 100.0% |
| vel_penalty | -8.656974 | -7.9% | 7.9% | 100.0% |
| contact_bonus | 3.300000 | 3.0% | 3.0% | 3.4% |
| orientation_cost | -3.182964 | -2.9% | 2.9% | 100.0% |
| landing_quality | 2.820021 | 2.6% | 2.6% | 1.5% |
| fuel_cost | -0.115500 | -0.1% | 0.1% | 5.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
