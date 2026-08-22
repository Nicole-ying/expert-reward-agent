# Training Feedback

## Final-policy outcome
score=204.383241, len=296.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[13.351261, 264.507293]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 1419.000000 | 91.6% | 91.6% | 27.8% |
| distance_penalty | -78.878156 | -5.1% | 5.1% | 100.0% |
| success_bonus | 43.500000 | 2.8% | 2.8% | 7.3% |
| vel_penalty | -4.294803 | -0.3% | 0.3% | 98.0% |
| angle_penalty | -3.326432 | -0.2% | 0.2% | 100.0% |
| angvel_penalty | -0.137135 | -0.0% | 0.0% | 94.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
