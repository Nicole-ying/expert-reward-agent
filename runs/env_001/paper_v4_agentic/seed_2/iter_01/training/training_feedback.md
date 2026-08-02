# Training Feedback

## Final-policy outcome
score=-116.462541, len=134.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-154.656939, -66.012571]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_delta | 0.872753 | 83.2% | 91.4% | 100.0% |
| velocity_penalty | -0.060163 | -5.7% | 5.7% | 1.5% |
| orientation_penalty | -0.030161 | -2.9% | 2.9% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
