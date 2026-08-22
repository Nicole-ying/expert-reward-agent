# Training Feedback

## Final-policy outcome
score=199.416737, len=477.650000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-41.000198, 281.710008]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 70.982707 | 79.7% | 79.7% | 32.3% |
| distance_progress | 6.303575 | 7.1% | 7.6% | 98.5% |
| descent_bonus | 6.376723 | 7.2% | 7.2% | 69.7% |
| fuel_penalty | -3.050500 | -3.4% | 3.4% | 63.9% |
| stability | -1.870730 | -2.1% | 2.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
