# Training Feedback

## Final-policy outcome
score=0.623359, len=117.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-81.298422, 226.110674]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| potential_diff | 1.073394 | 40.3% | 48.9% | 100.0% |
| angle_penalty | -1.151976 | -43.3% | 43.3% | 100.0% |
| landing_quality_reward | 0.208884 | 7.8% | 7.8% | 0.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
