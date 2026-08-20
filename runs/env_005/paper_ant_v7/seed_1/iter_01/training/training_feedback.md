# Training Feedback

## Final-policy outcome
score=2.911636, len=694.350000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-638.628762, 321.872725]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_gated | 728.613833 | 97.8% | 99.8% | 79.9% |
| action_penalty | -1.445550 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
