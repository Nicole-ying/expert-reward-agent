# Training Feedback

## Final-policy outcome
score=4.444416, len=17.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-21.057211, 21.900659]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 9.357899 | 53.7% | 53.7% | 86.5% |
| action_penalty | -4.869778 | -28.0% | 28.0% | 100.0% |
| lateral_penalty | -2.321523 | -13.3% | 13.3% | 100.0% |
| descend_penalty | -0.864603 | -5.0% | 5.0% | 42.2% |
| height_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
