# Training Feedback

## Final-policy outcome
score=266.046358, len=1143.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[15.918748, 290.488279]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity | 670.464700 | 69.3% | 69.3% | 100.0% |
| gait_alternation | 163.650000 | 16.9% | 16.9% | 71.5% |
| instability_penalty | -133.042173 | -13.8% | 13.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
