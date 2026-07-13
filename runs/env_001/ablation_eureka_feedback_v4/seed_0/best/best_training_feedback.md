# Training Feedback

## Final-policy outcome
score=239.517348, len=387.900000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[113.995736, 270.018272]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability | 382.118864 | 77.4% | 77.4% | 100.0% |
| contact | 96.029143 | 19.4% | 19.4% | 21.4% |
| engine_penalty | -8.973000 | -1.8% | 1.8% | 77.1% |
| progress | 6.637641 | 1.3% | 1.3% | 84.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
