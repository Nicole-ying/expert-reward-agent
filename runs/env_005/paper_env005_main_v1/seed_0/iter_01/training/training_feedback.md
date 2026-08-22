# Training Feedback

## Final-policy outcome
score=-123.547986, len=206.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-898.876280, 37.509336]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity_reward | 81.881057 | 50.7% | 71.9% | 100.0% |
| stability_penalty | -45.380735 | -28.1% | 28.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)
