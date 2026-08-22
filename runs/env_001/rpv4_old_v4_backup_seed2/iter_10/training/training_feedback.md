# Training Feedback

## Final-policy outcome
score=170.389671, len=789.500000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[85.445026, 251.420431]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_completion | 400.990792 | 95.0% | 95.0% | 26.4% |
| distance_progress | 6.683453 | 1.6% | 1.7% | 99.1% |
| descent_bonus | 7.325767 | 1.7% | 1.7% | 74.1% |
| fuel_penalty | -4.835500 | -1.1% | 1.1% | 61.2% |
| stability | -1.474795 | -0.3% | 0.3% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
