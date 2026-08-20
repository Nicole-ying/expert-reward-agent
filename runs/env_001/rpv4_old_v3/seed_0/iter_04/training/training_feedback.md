# Training Feedback

## Final-policy outcome
score=-56.068149, len=109.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-134.812423, 71.238886]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 2.592275 | 66.5% | 69.3% | 100.0% |
| landing_success_reward | 0.650000 | 16.7% | 16.7% | 0.6% |
| living_penalty | -0.546000 | -14.0% | 14.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 12/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
