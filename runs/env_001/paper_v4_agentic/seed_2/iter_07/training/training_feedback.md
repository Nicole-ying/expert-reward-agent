# Training Feedback

## Final-policy outcome
score=145.467167, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[115.762545, 185.132894]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| C_landing_steady | 197.840075 | 97.0% | 97.0% | 74.9% |
| A_progress_gated | 5.633926 | 2.8% | 3.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
