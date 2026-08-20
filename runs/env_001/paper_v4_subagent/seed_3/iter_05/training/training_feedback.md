# Training Feedback

## Final-policy outcome
score=142.839147, len=861.650000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-283.075416, 280.399777]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 65.745001 | 50.2% | 50.2% | 76.3% |
| landing_approach | 38.172721 | 29.1% | 29.1% | 100.0% |
| progress | 19.806749 | 15.1% | 16.2% | 99.9% |
| stability_penalty | -3.829228 | -2.9% | 2.9% | 100.0% |
| lateral_drift_penalty | -2.028130 | -1.5% | 1.5% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
