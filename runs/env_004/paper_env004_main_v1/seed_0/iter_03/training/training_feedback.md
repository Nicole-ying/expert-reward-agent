# Training Feedback

## Final-policy outcome
score=1338.039623, len=444.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[1199.694500, 3060.863104]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_stability_reward | 857.993966 | 90.2% | 90.2% | 100.0% |
| vertical_activity | 92.722003 | 9.7% | 9.7% | 100.0% |
| stability_penalty | -0.548786 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
