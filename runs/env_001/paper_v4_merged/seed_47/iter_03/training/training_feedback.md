# Training Feedback

## Final-policy outcome
score=-73.927368, len=815.100000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-208.636122, 263.001964]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_potential_diff | 4.114152 | 6.3% | 84.4% | 100.0% |
| contact_bonus | 3.000000 | 4.6% | 7.7% | 0.1% |
| progress_reward | 1.063202 | 1.6% | 4.7% | 99.9% |
| attitude_penalty | -2.065665 | -3.2% | 3.2% | 100.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
