# Training Feedback

## Final-policy outcome
score=-544.715736, len=647.800000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-1350.575507, 102.052073]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_upright_product | 281.574164 | 93.0% | 98.2% | 80.8% |
| height_survival_penalty | -5.414862 | -1.8% | 1.8% | 19.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
