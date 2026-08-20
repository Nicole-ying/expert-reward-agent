# Training Feedback

## Final-policy outcome
score=125.329636, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[79.069905, 161.303778]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 43.781036 | 71.3% | 71.3% | 100.0% |
| progress | 13.735301 | 22.4% | 23.7% | 100.0% |
| lateral_drift_penalty | -1.395423 | -2.3% | 2.3% | 100.0% |
| stability_penalty | -1.269045 | -2.1% | 2.1% | 100.0% |
| descending_penalty | -0.415939 | -0.7% | 0.7% | 1.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
