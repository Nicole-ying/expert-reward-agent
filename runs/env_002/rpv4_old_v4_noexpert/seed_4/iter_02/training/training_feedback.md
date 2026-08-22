# Training Feedback

## Final-policy outcome
score=274.342955, len=1427.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[54.500153, 295.210684]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 720.869569 | 88.7% | 88.8% | 100.0% |
| energy_efficiency_penalty | -47.934561 | -5.9% | 5.9% | 100.0% |
| upright_stability | -43.201623 | -5.3% | 5.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
