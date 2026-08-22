# Training Feedback

## Final-policy outcome
score=278.989576, len=1275.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[36.319760, 294.426659]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 697.414434 | 98.3% | 99.1% | 100.0% |
| upright_stability | -6.677646 | -0.9% | 0.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
