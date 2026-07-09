# Training Feedback

## Final-policy outcome
score=110.021266, len=613.350000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-28.784622, 289.902417]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_quality_reward | 493.419264 | 99.7% | 99.7% | 100.0% |
| progress_reward | 1.352045 | 0.3% | 0.3% | 99.9% |
| stability_penalty | -0.125754 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
