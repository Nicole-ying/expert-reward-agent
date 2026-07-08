# Training Feedback

## Final-policy outcome
score=-109.220612, len=68.550000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746612, -79.936904]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| weighted_stability_penalty | -25.740271 | -68.9% | 68.9% | 100.0% |
| progress_delta | 11.197361 | 30.0% | 31.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
