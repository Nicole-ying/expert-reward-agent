# Training Feedback

## Final-policy outcome
score=-110.221937, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.110856, -92.580119]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_delta | 56.005132 | 83.6% | 86.6% | 100.0% |
| velocity_danger | -8.552732 | -12.8% | 12.8% | 100.0% |
| orientation_penalty | -0.445298 | -0.7% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
