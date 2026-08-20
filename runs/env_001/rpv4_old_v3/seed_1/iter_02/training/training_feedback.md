# Training Feedback

## Final-policy outcome
score=-109.576065, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.894147, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.091045 | -94.3% | 94.3% | 100.0% |
| goal_approach | 1.120194 | 5.5% | 5.7% | 100.0% |
| crash_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
