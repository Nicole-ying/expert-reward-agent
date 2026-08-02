# Training Feedback

## Final-policy outcome
score=267.695456, len=289.000000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[242.123896, 298.738669]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 34.092490 | 81.2% | 81.2% | 100.0% |
| landing_event | 4.650000 | 11.1% | 11.6% | 0.3% |
| thrust_cost | -1.638000 | -3.9% | 3.9% | 56.7% |
| progress | 1.364806 | 3.3% | 3.3% | 94.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
