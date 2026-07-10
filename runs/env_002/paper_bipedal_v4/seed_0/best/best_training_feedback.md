# Training Feedback

## Final-policy outcome
score=300.224778, len=1078.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[299.070953, 301.485751]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity | 1008.781804 | 95.1% | 95.1% | 99.3% |
| stability | -43.264310 | -4.1% | 4.1% | 100.0% |
| energy | -8.251407 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
