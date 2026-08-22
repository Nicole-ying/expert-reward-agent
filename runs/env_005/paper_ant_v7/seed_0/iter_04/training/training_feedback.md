# Training Feedback

## Final-policy outcome
score=260.753814, len=421.000000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[7.919305, 699.515927]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 566.729879 | 70.1% | 72.4% | 100.0% |
| height_reward | 121.575984 | 15.0% | 15.0% | 100.0% |
| action_cost | -72.528575 | -9.0% | 9.0% | 100.0% |
| upright_penalty | -29.260389 | -3.6% | 3.6% | 15.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
