# Training Feedback

## Final-policy outcome
score=-79.484100, len=69.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-116.817045, -42.709771]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_progress | 11.152698 | 68.8% | 71.5% | 100.0% |
| landing_guidance | 0.646637 | 4.0% | 28.1% | 100.0% |
| tilt_penalty | -0.074832 | -0.5% | 0.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
