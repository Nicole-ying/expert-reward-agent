# Training Feedback

## Final-policy outcome
score=21.662903, len=610.450000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-206.440857, 239.571166]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_penalty | -269.646438 | -52.8% | 52.8% | 100.0% |
| landing_proxy | 229.469900 | 44.9% | 44.9% | 6.0% |
| orientation_penalty | -6.107955 | -1.2% | 1.2% | 100.0% |
| velocity_penalty | -5.444674 | -1.1% | 1.1% | 99.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
