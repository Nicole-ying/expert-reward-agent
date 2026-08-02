# Training Feedback

## Final-policy outcome
score=39.611787, len=211.250000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[12.124838, 89.968107]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.343811 | 78.2% | 81.5% | 100.0% |
| landing_safety_penalty | 0.317673 | 18.5% | 18.5% | 100.0% |
| x_boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
