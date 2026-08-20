# Training Feedback

## Final-policy outcome
score=243.728141, len=356.300000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[138.980073, 287.379143]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_progress | 145.720812 | 89.8% | 89.8% | 100.0% |
| distance_delta | 13.032984 | 8.0% | 8.5% | 96.3% |
| engine_penalty | -2.853500 | -1.8% | 1.8% | 80.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
