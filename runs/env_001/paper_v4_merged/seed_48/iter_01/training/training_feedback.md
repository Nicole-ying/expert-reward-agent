# Training Feedback

## Final-policy outcome
score=216.185314, len=554.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[167.347960, 261.244562]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing_progress | 314.216650 | 94.2% | 94.2% | 100.0% |
| distance_delta | 13.772733 | 4.1% | 4.3% | 97.4% |
| engine_penalty | -5.176000 | -1.6% | 1.6% | 93.4% |
| angle_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
