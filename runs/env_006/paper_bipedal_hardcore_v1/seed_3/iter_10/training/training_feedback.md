# Training Feedback

## Final-policy outcome
score=-66.619855, len=167.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-85.657212, -39.056887]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 51.491098 | 80.0% | 80.0% | 97.0% |
| air_penalty | -12.857803 | -20.0% | 20.0% | 95.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
