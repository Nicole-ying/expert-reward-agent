# Training Feedback

## Final-policy outcome
score=-34.070036, len=550.550000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-170.693540, 186.842255]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 392.604940 | 62.2% | 62.2% | 100.0% |
| distance_reward | -208.890941 | -33.1% | 33.1% | 100.0% |
| stability_penalty | -29.893415 | -4.7% | 4.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
