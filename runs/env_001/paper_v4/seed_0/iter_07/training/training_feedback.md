# Training Feedback

## Final-policy outcome
score=-388.671871, len=109.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-499.654146, -304.708700]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -3.597500 | -48.3% | 48.3% | 65.7% |
| descent_safety | -2.805849 | -37.7% | 37.7% | 6.2% |
| approach_shaping | -0.933721 | -12.5% | 14.0% | 100.0% |
| stable_landed | 0.000000 | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
